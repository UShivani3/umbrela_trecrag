from functools import lru_cache
import json
import os
import random
import re
import platform
import subprocess

try:
    from pyserini.index.lucene import LuceneIndexReader
except ImportError:
    print("\nPyserini version is likely too old, check and ensure pyserini >= 0.1.2.0\n")
    
from pyserini.search import get_qrels_file, get_topics


def get_catwise_data(qrel_data, few_shot_count):
    """Extract category-wise data for few-shot examples."""
    examples = []
    for cat in [0, 1, 2, 3]:
        req_tuple_list = []

        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    req_tuple_list.append((qid, doc_id))
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")

        assert (
            len(req_tuple_list) >= few_shot_count
        ), f"Count of judgments available for category {cat} is lesser than {few_shot_count}."

        if len(req_tuple_list):
            samples_for_examples = random.sample(req_tuple_list, few_shot_count)
            examples.extend(samples_for_examples)
    return examples


def examples_prompt(few_shot_examples, query_mappings, passage_retriever_type, 
                   qrel_data, **passage_retriever_kwargs):
    """Generate prompt examples for few-shot learning."""
    prompt_examples = ""

    for example in few_shot_examples:
        # Handle both string and integer query IDs
        qid = example[0]
        if isinstance(qid, int) or qid.isdigit():
            query_key = int(qid)
        else:
            query_key = qid
            
        query = query_mappings[query_key]["title"]
        passage = get_passage_wrapper(
            corpus_identifier=None, 
            doc_id=example[1],
            passage_retriever_type=passage_retriever_type,
            **passage_retriever_kwargs
        )

        res_json = f"##final score: {int(qrel_data[example[0]][example[1]])}"
        prompt_examples += f"""
            ###

            Query: {query}
            Passage: {passage}
            {res_json}
            """
    return prompt_examples


def get_query_mappings(qrel=None, query_mapping_file=None, custom_query_mappings=None,
                      query_mapping_format="auto"):
    """
    Get query mappings from various sources with flexible format support.
    
    Args:
        qrel: Standard TREC qrel identifier
        query_mapping_file: Path to file with query mappings (JSON/TSV)
        custom_query_mappings: Dictionary with custom query mappings
        query_mapping_format: Format of query_mapping_file:
            - "auto": Auto-detect from file extension
            - "json": JSON format {"qid": {"title": "query", ...}, ...}
            - "json_simple": Simple JSON format {"qid": "query", ...}
            - "tsv": TSV format with columns: qid\ttitle
            - "pyserini": Use Pyserini's get_topics (for standard TREC datasets)
    
    Returns:
        Dictionary mapping query IDs to query information
    """
    if custom_query_mappings:
        return custom_query_mappings
    elif qrel:
        # Query mappings for standard TREC datasets using Pyserini
        topic_mapping = {
            "dl19-passage": "dl19-passage",
            "dl20-passage": "dl20-passage", 
            "dl21-passage": "dl21",
            "dl22-passage": "dl22",
            "dl23-passage": "dl23",
            "robust04": "robust04",
            "robust05": "robust05",
        }
        if qrel not in topic_mapping:
            raise ValueError(f"Invalid value for qrel: {qrel}. Use custom_query_mappings for custom datasets.")
        query_mappings = get_topics(topic_mapping[qrel])
    elif query_mapping_file:
        # Auto-detect format if not specified
        if query_mapping_format == "auto":
            file_ext = os.path.splitext(query_mapping_file)[1].lower()
            if file_ext == ".json":
                query_mapping_format = "json"
            elif file_ext == ".tsv" or file_ext == ".txt":
                query_mapping_format = "tsv"
            else:
                query_mapping_format = "json"  # Default fallback
        
        query_mappings = load_query_mappings_from_file(query_mapping_file, query_mapping_format)
    else:
        raise ValueError("Must provide either qrel, query_mapping_file, or custom_query_mappings")
    
    return query_mappings


def generate_examples_prompt(qrel=None, few_shot_count=0, custom_qrel_path=None, 
                           query_mapping_file=None, custom_query_mappings=None,
                           query_mapping_format="auto", passage_retriever_type=None,
                           corpus_identifier=None, **passage_retriever_kwargs):
    """
    Generate few-shot examples prompt for both standard and custom qrels.
    
    Args:
        qrel: Standard TREC qrel identifier (e.g., 'dl19-passage')
        few_shot_count: Number of examples per category
        custom_qrel_path: Path to custom qrel file
        query_mapping_file: Path to query mappings file
        custom_query_mappings: Dictionary with custom query mappings
        query_mapping_format: Format of query mapping file (auto, json, json_simple, tsv)
        passage_retriever_type: Type of passage retriever to use
        corpus_identifier: [DEPRECATED] Use passage_retriever_type instead
        **passage_retriever_kwargs: Additional arguments for passage retrieval
    
    Returns:
        String containing formatted prompt examples
    """
    # Determine qrel source
    qrel_source = custom_qrel_path if custom_qrel_path else qrel
    if not qrel_source:
        raise ValueError("Must provide either qrel or custom_qrel_path")
    
    # Determine passage retriever type
    if not passage_retriever_type:
        if corpus_identifier:
            # Backward compatibility mapping
            if corpus_identifier == "msmarcov1":
                passage_retriever_type = "pyserini_msmarco_v1"
            elif corpus_identifier == "msmarcov2":
                passage_retriever_type = "msmarco_v2_files"
            elif corpus_identifier == "custom":
                passage_retriever_type = "custom_function"
        elif qrel:
            passage_retriever_type = infer_passage_retriever_type(qrel)
        else:
            raise ValueError("Must specify passage_retriever_type or provide qrel for auto-inference")
    
    qrel_data = get_qrels(qrel_source)
    few_shot_examples = get_catwise_data(qrel_data, few_shot_count)
    query_mappings = get_query_mappings(qrel, query_mapping_file, custom_query_mappings, query_mapping_format)
    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, passage_retriever_type, qrel_data, **passage_retriever_kwargs
    )
    return prompt_examples


def generate_holes(qrel=None, judge_cat=[0, 1, 2, 3], exception_qid=[], custom_qrel_path=None):
    """
    Generate evaluation holes for both standard and custom qrels.
    
    Args:
        qrel: Standard TREC qrel identifier
        judge_cat: Categories to include in evaluation
        exception_qid: Query IDs to exclude
        custom_qrel_path: Path to custom qrel file
    
    Returns:
        Tuple of (holes, ground_truths)
    """
    qrel_source = custom_qrel_path if custom_qrel_path else qrel
    if not qrel_source:
        raise ValueError("Must provide either qrel or custom_qrel_path")
        
    qrel_data = get_qrels(qrel_source)
    holes = []
    gts = []
    for cat in judge_cat:
        req_tuple_list = []

        total_count = 0
        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    total_count += 1
                    if qid not in exception_qid:
                        req_tuple_list.append((qid, doc_id))

        samples = req_tuple_list
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")
        holes += samples
        gts += [cat] * len(samples)
    return holes, gts


def get_qrel_path(qrel_info):
    """Get the path to qrel file, handling both standard and custom qrels."""
    if not os.path.exists(qrel_info):
        # Try to get standard TREC qrel
        try:
            return get_qrels_file(qrel_info)
        except:
            raise FileNotFoundError(f"Qrel file not found: {qrel_info}")
    return qrel_info


def get_qrels(qrel_info):
    """Load qrel data from file (modified version of pyserini's get_qrels)."""
    file_path = get_qrel_path(qrel_info)

    qrels = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 4:
                continue
                
            qid, _, docid, judgement = parts[:4]

            # Handle both string and numeric IDs
            if qid.isdigit():
                qrels_key = int(qid)
            else:
                qrels_key = qid

            if docid.isdigit():
                doc_key = int(docid)
            else:
                doc_key = docid

            if qrels_key in qrels:
                qrels[qrels_key][doc_key] = judgement
            else:
                qrels[qrels_key] = {doc_key: judgement}
    return qrels


def infer_passage_retriever_type(qrel):
    """Infer passage retriever type from standard qrel names."""
    if not qrel:
        return None
        
    if qrel in ["dl19-passage", "dl20-passage"]:
        return "pyserini_msmarco_v1"
    elif qrel in ["dl21", "dl22", "dl23"]:
        return "msmarco_v2_files"
    elif qrel in ["robust04", "robust05"]:
        return "pyserini_custom"  # Users need to specify index_path
    else:
        return "custom_function"


def get_passage_msv2(pid):
    """Get passage from MS MARCO v2 format."""
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(
        f"../data/msmarco_v2_passage/msmarco_passage_{bundlenum}", "rt", encoding="utf8"
    ) as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid
        return document["passage"]


def prepare_query_passage(qid_docid_list, passage_retriever_type=None, query_mappings=None, 
                         qrel=None, query_mapping_file=None, custom_query_mappings=None,
                         query_mapping_format="auto", corpus_identifier=None, 
                         **passage_retriever_kwargs):
    """
    Prepare query-passage pairs for evaluation.
    
    Args:
        qid_docid_list: List of (query_id, doc_id) tuples
        passage_retriever_type: Type of passage retriever to use
        query_mappings: Pre-loaded query mappings
        qrel: Standard TREC qrel identifier
        query_mapping_file: Path to query mappings file
        custom_query_mappings: Custom query mappings dictionary
        query_mapping_format: Format of query mapping file
        corpus_identifier: [DEPRECATED] Use passage_retriever_type instead
        **passage_retriever_kwargs: Additional arguments for passage retrieval
    
    Returns:
        List of (query_text, passage_text) tuples
    """
    if not query_mappings:
        query_mappings = get_query_mappings(qrel, query_mapping_file, custom_query_mappings, query_mapping_format)
    
    # Determine passage retriever type
    if not passage_retriever_type:
        if corpus_identifier:
            # Backward compatibility
            if corpus_identifier == "msmarcov1":
                passage_retriever_type = "pyserini_msmarco_v1"
            elif corpus_identifier == "msmarcov2":
                passage_retriever_type = "msmarco_v2_files"
            elif corpus_identifier == "custom":
                passage_retriever_type = "custom_function"
        elif qrel:
            passage_retriever_type = infer_passage_retriever_type(qrel)
        else:
            raise ValueError("Must specify passage_retriever_type")
    
    query_passage = []
    for sample in qid_docid_list:
        qid = sample[0]
        doc_id = sample[1]
        
        # Handle both string and integer query IDs
        if isinstance(qid, int) or (isinstance(qid, str) and qid.isdigit()):
            query_key = int(qid)
        else:
            query_key = qid
            
        passage = get_passage_wrapper(
            corpus_identifier=None,
            doc_id=doc_id, 
            passage_retriever_type=passage_retriever_type,
            **passage_retriever_kwargs
        )
        query_text = query_mappings[query_key]["title"]
        query_passage.append((query_text, passage))
    
    return query_passage


# Example usage functions for custom datasets
def create_custom_query_mappings(queries_dict):
    """
    Helper function to create query mappings in the expected format.
    
    Args:
        queries_dict: Dictionary mapping query_id -> query_text
    
    Returns:
        Dictionary in the format expected by other functions
    """
    return {qid: {"title": query_text} for qid, query_text in queries_dict.items()}


def custom_passage_retriever_example(doc_id):
    """
    Example custom passage retriever function.
    Users should implement their own based on their corpus format.
    
    Args:
        doc_id: Document identifier
    
    Returns:
        Passage text
    """
    # This is just an example - replace with your actual retrieval logic
    # e.g., reading from a JSON file, database, or API
    raise NotImplementedError("Implement your custom passage retrieval logic here")

def get_passage_wrapper(corpus_identifier, doc_id, passage_retriever_type=None, 
                       custom_passage_retriever=None, passage_file_path=None,
                       index_path=None):
    """
    Wrapper function to retrieve passages from different corpora with flexible retrieval options.
    
    Args:
        corpus_identifier: Identifier for the corpus type (deprecated, use passage_retriever_type)
        doc_id: Document ID
        passage_retriever_type: Type of passage retriever to use:
            - "pyserini_msmarco_v1": Use Pyserini index for MS MARCO v1
            - "pyserini_msmarco_v2": Use Pyserini index for MS MARCO v2  
            - "pyserini_custom": Use custom Pyserini index
            - "msmarco_v2_files": Read from MS MARCO v2 passage files
            - "json_file": Read from JSON file containing passages
            - "tsv_file": Read from TSV file containing passages
            - "custom_function": Use custom retrieval function
        custom_passage_retriever: Custom function for passage retrieval
        passage_file_path: Path to passage file (for json_file, tsv_file types)
        index_path: Path to custom Pyserini index (for pyserini_custom type)
    
    Returns:
        String containing the passage text
    """
    # Backward compatibility: infer passage_retriever_type from corpus_identifier
    if not passage_retriever_type and corpus_identifier:
        if corpus_identifier == "msmarcov1":
            passage_retriever_type = "pyserini_msmarco_v1"
        elif corpus_identifier == "msmarcov2":
            passage_retriever_type = "msmarco_v2_files"
        elif corpus_identifier == "custom":
            passage_retriever_type = "custom_function"
    
    if passage_retriever_type == "pyserini_msmarco_v1":
        index_reader = LuceneIndexReader.from_prebuilt_index("msmarco-v1-passage")
        passage = json.loads(index_reader.doc_raw(str(doc_id))).get("contents", "")
        
    elif passage_retriever_type == "pyserini_msmarco_v2":
        index_reader = LuceneIndexReader.from_prebuilt_index("msmarco-v2-passage")
        passage = json.loads(index_reader.doc_raw(str(doc_id))).get("contents", "")
        
    elif passage_retriever_type == "pyserini_custom":
        if not index_path:
            raise ValueError("index_path must be provided for pyserini_custom type")
        index_reader = LuceneIndexReader(index_path)
        passage = json.loads(index_reader.doc_raw(str(doc_id))).get("contents", "")
        
    elif passage_retriever_type == "msmarco_v2_files":
        passage = get_passage_msv2(doc_id)
        
    elif passage_retriever_type == "json_file":
        if not passage_file_path:
            raise ValueError("passage_file_path must be provided for json_file type")
        passage = get_passage_from_json(doc_id, passage_file_path)
        
    elif passage_retriever_type == "custom_function":
        if not custom_passage_retriever:
            raise ValueError("custom_passage_retriever must be provided for custom_function type")
        passage = custom_passage_retriever(doc_id)
        
    else:
        raise ValueError(f"Unsupported passage_retriever_type: {passage_retriever_type}")
    
    return passage


def prepare_query_passage(qid_docid_list, corpus_identifier, query_mappings=None, 
                         qrel=None, query_mapping_file=None, custom_query_mappings=None,
                         custom_passage_retriever=None):
    """
    Prepare query-passage pairs for evaluation.
    
    Args:
        qid_docid_list: List of (query_id, doc_id) tuples
        corpus_identifier: Corpus identifier for passage retrieval
        query_mappings: Pre-loaded query mappings
        qrel: Standard TREC qrel identifier
        query_mapping_file: Path to query mappings file
        custom_query_mappings: Custom query mappings dictionary
        custom_passage_retriever: Custom passage retrieval function
    
    Returns:
        List of (query_text, passage_text) tuples
    """
    if not query_mappings:
        query_mappings = get_query_mappings(qrel, query_mapping_file, custom_query_mappings)
    
    query_passage = []
    for sample in qid_docid_list:
        qid = sample[0]
        doc_id = sample[1]
        
        # Handle both string and integer query IDs
        if isinstance(qid, int) or (isinstance(qid, str) and qid.isdigit()):
            query_key = int(qid)
        else:
            query_key = qid
            
        passage = get_passage_wrapper(corpus_identifier, doc_id, custom_passage_retriever)
        query_text = query_mappings[query_key]["title"]
        query_passage.append((query_text, passage))
    
    return query_passage


def fetch_ndcg_score(qrel_path, result_path):
    """Fetch NDCG score using trec_eval."""
    cmd = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {qrel_path} {result_path}"
    cmd = cmd.split(" ")
    shell = platform.system() == "Windows"
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    stdout, stderr = process.communicate()
    output = stdout.decode("utf-8").rstrip()
    print(output)

    pattern = r"ndcg_cut_10\s+all\s+([0-9.]+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        return 0


def get_dropped_cat_count(qrel=None, removal_fraction=0.0, custom_qrel_path=None):
    """
    Get count of judgments after dropping a fraction from each category.
    
    Args:
        qrel: Standard TREC qrel identifier
        removal_fraction: Fraction of judgments to remove (0.0 to 1.0)
        custom_qrel_path: Path to custom qrel file
    
    Returns:
        Dictionary mapping categories to remaining judgment counts
    """
    qrel_source = custom_qrel_path if custom_qrel_path else qrel
    if not qrel_source:
        raise ValueError("Must provide either qrel or custom_qrel_path")
        
    qrel_data = get_qrels(qrel_source)

    cat_dict = {}
    for index, cat in enumerate([0, 1, 2, 3]):
        req_tuple_list = []

        total_count = 0
        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    total_count += 1
                    req_tuple_list.append((qid, doc_id))

        remaining_count = len(req_tuple_list) - int(len(req_tuple_list) * removal_fraction)
        print(
            f"No. of judgments for category {cat}: {len(req_tuple_list)}. "
            f"Judgments that remain intact: {remaining_count}"
        )
        cat_dict[str(cat)] = remaining_count
        
    return cat_dict


# New helper functions for different file formats and passage retrieval methods

def load_query_mappings_from_file(file_path, format_type):
    """
    Load query mappings from various file formats.
    
    Args:
        file_path: Path to the query mappings file
        format_type: Format of the file (json, json_simple, tsv)
    
    Returns:
        Dictionary mapping query IDs to query information
    """
    if format_type == "json":
        with open(file_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    
    elif format_type == "json_simple":
        with open(file_path, "r", encoding="utf-8") as fp:
            simple_mappings = json.load(fp)
            # Convert simple format to expected format
            return {qid: {"title": query_text} for qid, query_text in simple_mappings.items()}
    
    elif format_type == "tsv":
        mappings = {}
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qid = parts[0]
                    if qid.isdigit():
                        qid = int(qid)
                    query_text = parts[1]
                    # Handle additional fields if present
                    description = parts[2] if len(parts) > 2 else ""
                    narrative = parts[3] if len(parts) > 3 else ""
                    
                    mappings[qid] = {
                        "title": query_text,
                        "description": description,
                        "narrative": narrative
                    }
        return mappings
    
    else:
        raise ValueError(f"Unsupported query mapping format: {format_type}")


@lru_cache(maxsize=10000)
def get_passage_from_json(doc_id, json_file_path):
    """
    Retrieve passage from JSON file.
    Expected format: {"doc_id": "passage_text", ...} or 
                    {"doc_id": {"text": "passage_text", ...}, ...}
    
    Args:
        doc_id: Document ID
        json_file_path: Path to JSON file containing passages
    
    Returns:
        Passage text
    """
    if not hasattr(get_passage_from_json, '_cache'):
        get_passage_from_json._cache = {}
    
    if json_file_path not in get_passage_from_json._cache:
        with open(json_file_path, "r", encoding="utf-8") as fp:
            get_passage_from_json._cache[json_file_path] = json.load(fp)
    
    passages = get_passage_from_json._cache[json_file_path]
    
    # Convert doc_id to string for lookup
    doc_key = str(doc_id)
    if doc_key in passages:
        passage_data = passages[doc_key]
        if isinstance(passage_data, str):
            return passage_data
        elif isinstance(passage_data, dict):
            # Try common field names
            return passage_data.get("text", passage_data.get("passage", passage_data.get("content", "")))
    
    # Try integer key if string key doesn't work
    if isinstance(doc_id, str) and doc_id.isdigit():
        int_key = int(doc_id)
        if int_key in passages:
            passage_data = passages[int_key]
            if isinstance(passage_data, str):
                return passage_data
            elif isinstance(passage_data, dict):
                return passage_data.get("text", passage_data.get("passage", passage_data.get("content", "")))
    
    raise KeyError(f"Document ID {doc_id} not found in {json_file_path}")


def get_available_passage_retrievers():
    """
    Get list of available passage retriever types.
    
    Returns:
        Dictionary with retriever types and their descriptions
    """
    return {
        "pyserini_msmarco_v1": "Use Pyserini prebuilt index for MS MARCO v1 passages",
        "pyserini_msmarco_v2": "Use Pyserini prebuilt index for MS MARCO v2 passages", 
        "pyserini_custom": "Use custom Pyserini index (requires index_path)",
        "msmarco_v2_files": "Read from MS MARCO v2 passage bundle files",
        "json_file": "Read passages from JSON file (requires passage_file_path)",
        "tsv_file": "Read passages from TSV file (requires passage_file_path)", 
        "custom_function": "Use custom retrieval function (requires custom_passage_retriever)"
    }