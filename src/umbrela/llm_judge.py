from abc import ABC, abstractmethod
import pkg_resources
import os
import statistics
import time

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from umbrela.utils import qrel_utils, common_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel: str = None,
        corpus: str = None,
        query_mapping_file: str = None,
        model_name: str = None,
        prompt_file: str = None,
        prompt_type: str = None,
        few_shot_count: int = 0,
        # New parameters for enhanced qrel support
        custom_qrel_path: str = None,
        custom_query_mappings: dict = None,
        query_mapping_format: str = "auto",
        passage_retriever_type: str = None,
        **passage_retriever_kwargs
    ) -> None:
        assert not (
            prompt_file and prompt_type
        ), "Both prompt_file and prompt_type passed. Only one mode must be selected!!"

        # Validate qrel source
        if not qrel and not custom_qrel_path:
            raise ValueError("Must provide either qrel or custom_qrel_path")

        self.qrel = qrel
        self.custom_qrel_path = custom_qrel_path
        self.corpus = corpus
        self.query_mapping_file = query_mapping_file
        self.custom_query_mappings = custom_query_mappings
        self.query_mapping_format = query_mapping_format
        self.passage_retriever_type = passage_retriever_type
        self.passage_retriever_kwargs = passage_retriever_kwargs
        self.few_shot_count = few_shot_count

        # Handle prompt configuration
        if prompt_type:
            if prompt_type not in ["bing", "basic"]:
                raise ValueError(f"Invalid prompt_type: {prompt_type}.")
            prompt_mode_str = "fewshot" if few_shot_count > 0 else "zeroshot"
            prompt_file = pkg_resources.resource_filename(
                "umbrela", f"prompts/qrel_{prompt_mode_str}_{prompt_type}.txt"
            )
            if not os.path.exists(prompt_file):
                raise ValueError(f"Prompt file doesn't exist.")

        if prompt_file:
            print(
                "Warning!! Prompt file expects input fields namely: (examples, query, passage)."
            )
        
        self.model_name = model_name
        
        # Generate few-shot examples using enhanced qrel_utils
        if few_shot_count > 0:
            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel=qrel,
                few_shot_count=few_shot_count,
                custom_qrel_path=custom_qrel_path,
                query_mapping_file=query_mapping_file,
                custom_query_mappings=custom_query_mappings,
                query_mapping_format=query_mapping_format,
                passage_retriever_type=passage_retriever_type,
                corpus_identifier=corpus,  # For backward compatibility
                **passage_retriever_kwargs
            )
        elif few_shot_count == 0:
            self.prompt_examples = ""
            if prompt_file and "fewshot" in prompt_file:
                print(
                    f"Warning!! default fewshot prompt file being used for few_shot_count = 0"
                )
        else:
            raise ValueError(f"Invalid value for few_shot_count: {few_shot_count}")

        with open(prompt_file) as p:
            self._prompt_template = "".join(p.readlines()).strip()

    def display_prompt_template(self):
        print(self._prompt_template)

    @abstractmethod
    def predict_with_llm(self, request_dict, max_new_tokens, prepocess):
        pass

    @abstractmethod
    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        pass

    def calculate_kappa(self, gts, preds):
        print(f"Kohen kappa overall: {cohen_kappa_score(gts, preds)}")
        print("-" * 79)
        gts_bin = [1 if int(x) > 1 else 0 for x in gts]
        preds_bin = [1 if int(x) > 1 else 0 for x in preds]
        print(f"Binarized Kohen kappa overall: {cohen_kappa_score(gts_bin, preds_bin)}")
        print("-" * 79)

    def draw_confusion_matrix(self, gts, preds):
        conf_mat = confusion_matrix(gts, preds)
        print(conf_mat)

        os.makedirs("conf_matrix", exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap="GnBu")
        for text in disp.text_.ravel():
            text.set_fontsize(16)
        
        # Use appropriate identifier for title
        title_identifier = self.qrel or os.path.basename(self.custom_qrel_path or "custom")
        ax.set_title(title_identifier, fontsize=14)
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)
        
        # Use appropriate model name for filename
        model_basename = os.path.basename(self.model_name) if self.model_name else "unknown_model"
        plt.savefig(f"conf_matrix/{title_identifier}-{model_basename}.png")

    def get_qrel_source(self):
        """Get the qrel source (either standard qrel or custom qrel path)."""
        return self.custom_qrel_path if self.custom_qrel_path else self.qrel

    def evalute_results_with_qrel(
        self,
        result_file,
        judge_cat=[0, 1, 2, 3],
        regenerate=False,
        num_samples=1,
        return_results_path=False,
    ):
        result_dir = f"modified_qrels"
        os.makedirs(result_dir, exist_ok=True)

        # Handle both standard and custom qrels
        qrel_source = self.get_qrel_source()
        path = qrel_utils.get_qrel_path(qrel_source)
        
        # Create appropriate model identifier
        model_identifier = self.model_name.split('/')[-1] if self.model_name else "unknown_model"
        qrel_identifier = self.qrel or os.path.basename(self.custom_qrel_path or "custom")
        
        modified_qrel = f"{result_dir}/{os.path.basename(path)[:-4]}_{model_identifier}_{''.join(map(str, judge_cat))}_{self.few_shot_count}_{num_samples}.txt"
        print(f"Output file: {modified_qrel}")

        if os.path.exists(modified_qrel) and not regenerate:
            # Load qrels using enhanced method
            org_qd = qrel_utils.get_qrels(qrel_source)
            new_qd = qrel_utils.get_qrels(modified_qrel)

            unmatch_dict = {}
            gts, preds = [], []

            for qid in org_qd:
                for docid in org_qd[qid]:
                    if org_qd[qid][docid] not in unmatch_dict:
                        unmatch_dict[org_qd[qid][docid]] = []
                    unmatch_dict[org_qd[qid][docid]].append(
                        int(org_qd[qid][docid] == new_qd[qid][docid])
                    )
                    gts.append(org_qd[qid][docid])
                    preds.append(new_qd[qid][docid])

        else:
            # Generate holes using enhanced method
            holes_tup, gts = qrel_utils.generate_holes(
                qrel=self.qrel, 
                judge_cat=judge_cat,
                custom_qrel_path=self.custom_qrel_path
            )
            
            qrel_data = qrel_utils.get_qrels(qrel_source)
            unmatch_dict = {}
            
            # Prepare query-passage pairs using enhanced method
            holes_qp = qrel_utils.prepare_query_passage(
                holes_tup, 
                passage_retriever_type=self.passage_retriever_type,
                qrel=self.qrel,
                query_mapping_file=self.query_mapping_file,
                custom_query_mappings=self.custom_query_mappings,
                query_mapping_format=self.query_mapping_format,
                corpus_identifier=self.corpus,  # For backward compatibility
                **self.passage_retriever_kwargs
            )
            
            if num_samples > 1:
                holes_qp = [item for item in holes_qp for _ in range(num_samples)]
                holes_tup = [item for item in holes_tup for _ in range(num_samples)]
                gts = [item for item in gts for _ in range(num_samples)]

            judgments = self.judge(holes_qp, prepocess=False, max_new_tokens=200)

            valid_res = {}
            preds = []
            gts_valid, preds_valid = [], []
            for index in range(0, len(judgments), num_samples):
                temp = []
                for internal_index in range(index, index + num_samples):
                    gt = gts[internal_index]
                    judgment = judgments[internal_index]
                    preds.append(judgment["judgment"])
                    curr_res = int(gt == judgment["judgment"])
                    temp.append(judgment["judgment"])
                    if gt not in unmatch_dict:
                        unmatch_dict[gt] = [curr_res]
                    else:
                        unmatch_dict[gt].append(curr_res)
                    if judgment["result_status"]:
                        gts_valid.append(gt)
                        preds_valid.append(judgment["judgment"])
                        if gt not in valid_res:
                            valid_res[gt] = [curr_res]
                        else:
                            valid_res[gt].append(curr_res)
                pair = holes_tup[index]
                qrel_data[pair[0]][pair[1]] = int(statistics.mode(temp))

            common_utils.write_modified_qrel(qrel_data, modified_qrel)
            print("For valid results:")
            self.calculate_kappa(gts_valid, preds_valid)
            for cat in valid_res:
                print(
                    f"Stats for {cat}. Correct judgments count in valid result: {sum(valid_res[cat])}/{len(valid_res[cat])}"
                )

        print("For overall results:")
        self.calculate_kappa(gts, preds)
        self.draw_confusion_matrix(gts, preds)

        for cat in unmatch_dict:
            print(
                f"Stats for {cat}. Correct judgments count: {sum(unmatch_dict[cat])}/{len(unmatch_dict[cat])}"
            )

        if result_file:
            print("-" * 79)
            output = {}
            # Use enhanced fetch_ndcg_score method
            output["original"] = qrel_utils.fetch_ndcg_score(qrel_source, result_file)
            output[f"modified"] = qrel_utils.fetch_ndcg_score(modified_qrel, result_file)
            print(output)

        if return_results_path:
            return modified_qrel

    def get_dropped_cat_count(self, removal_fraction=0.0):
        """
        Get count of judgments after dropping a fraction from each category.
        Uses enhanced qrel_utils method.
        """
        return qrel_utils.get_dropped_cat_count(
            qrel=self.qrel,
            removal_fraction=removal_fraction,
            custom_qrel_path=self.custom_qrel_path
        )

    def get_available_passage_retrievers(self):
        """Get list of available passage retriever types."""
        return qrel_utils.get_available_passage_retrievers()

    @classmethod
    def create_with_custom_data(
        cls,
        custom_qrel_path: str,
        query_mappings: dict,
        passage_retriever_type: str,
        model_name: str,
        prompt_type: str = "basic",
        few_shot_count: int = 1,
        **passage_retriever_kwargs
    ):
        """
        Convenience method to create LLMJudge with custom data.
        
        Args:
            custom_qrel_path: Path to custom qrel file
            query_mappings: Dictionary mapping query_id -> query_text
            passage_retriever_type: Type of passage retriever
            model_name: Name/path of the model
            prompt_type: Type of prompt ("basic" or "bing")
            few_shot_count: Number of few-shot examples
            **passage_retriever_kwargs: Additional args for passage retriever
        
        Returns:
            LLMJudge instance configured for custom data
        """
        # Convert simple query mappings to expected format
        if query_mappings and isinstance(list(query_mappings.values())[0], str):
            query_mappings = qrel_utils.create_custom_query_mappings(query_mappings)
        
        return cls(
            custom_qrel_path=custom_qrel_path,
            custom_query_mappings=query_mappings,
            passage_retriever_type=passage_retriever_type,
            model_name=model_name,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
            **passage_retriever_kwargs
        )

    @classmethod  
    def create_with_files(
        cls,
        custom_qrel_path: str,
        query_mapping_file: str,
        passage_file_path: str,
        model_name: str,
        query_mapping_format: str = "auto",
        passage_retriever_type: str = "auto",
        prompt_type: str = "basic",
        few_shot_count: int = 1
    ):
        """
        Convenience method to create LLMJudge with file-based data.
        
        Args:
            custom_qrel_path: Path to custom qrel file
            query_mapping_file: Path to query mappings file
            passage_file_path: Path to passage file
            model_name: Name/path of the model
            query_mapping_format: Format of query mapping file
            passage_retriever_type: Type of passage retriever (auto-detect if "auto")
            prompt_type: Type of prompt ("basic" or "bing")
            few_shot_count: Number of few-shot examples
        
        Returns:
            LLMJudge instance configured for file-based data
        """
        # Auto-detect passage retriever type from file extension
        if passage_retriever_type == "auto":
            file_ext = os.path.splitext(passage_file_path)[1].lower()
            if file_ext == ".json":
                passage_retriever_type = "json_file"
            elif file_ext in [".tsv", ".txt"]:
                passage_retriever_type = "tsv_file"
            else:
                passage_retriever_type = "json_file"  # Default fallback
        
        return cls(
            custom_qrel_path=custom_qrel_path,
            query_mapping_file=query_mapping_file,
            query_mapping_format=query_mapping_format,
            passage_retriever_type=passage_retriever_type,
            passage_file_path=passage_file_path,
            model_name=model_name,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count
        )


# Example usage for different scenarios
def example_usage():
    """Examples of how to use the enhanced LLMJudge class."""
    
    # 1. Standard TREC dataset (backward compatibility)
    judge_standard = LLMJudge(
        qrel="dl19-passage",
        corpus="msmarcov1",  # This will be mapped to passage_retriever_type
        model_name="gpt-3.5-turbo",
        prompt_type="basic",
        few_shot_count=2
    )
    
    # 2. Custom dataset with direct mappings
    custom_queries = {1: "What is AI?", 2: "How does ML work?"}
    judge_custom = LLMJudge.create_with_custom_data(
        custom_qrel_path="my_qrels.txt",
        query_mappings=custom_queries,
        passage_retriever_type="json_file",
        passage_file_path="my_passages.json",
        model_name="claude-3-sonnet",
        few_shot_count=1
    )
    
    # 3. Custom dataset with file-based configuration
    judge_files = LLMJudge.create_with_files(
        custom_qrel_path="my_qrels.txt",
        query_mapping_file="queries.tsv",
        passage_file_path="passages.tsv",
        model_name="gpt-4",
        query_mapping_format="tsv",
        passage_retriever_type="tsv_file"
    )
    
    # 4. Custom dataset with Pyserini index
    judge_pyserini = LLMJudge(
        custom_qrel_path="my_qrels.txt",
        query_mapping_file="queries.json",
        passage_retriever_type="pyserini_custom",
        index_path="/path/to/my/index",
        model_name="llama-2-70b",
        prompt_type="bing",
        few_shot_count=3
    )