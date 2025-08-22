import argparse
from typing_extensions import Optional
from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams
from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class OSLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str = None,
        model_name: str = None,
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 2,
        device: str = "cuda",
        num_gpus: int = 1,
        corpus: str = None,
        query_mapping_file: str = None,
        # New parameters for enhanced qrel support
        custom_qrel_path: str = None,
        custom_query_mappings: dict = None,
        query_mapping_format: str = "auto",
        passage_retriever_type: str = None,
        **passage_retriever_kwargs
    ) -> None:
        super().__init__(
            qrel=qrel,
            corpus=corpus,
            query_mapping_file=query_mapping_file,
            model_name=model_name,
            prompt_file=prompt_file,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
            custom_qrel_path=custom_qrel_path,
            custom_query_mappings=custom_query_mappings,
            query_mapping_format=query_mapping_format,
            passage_retriever_type=passage_retriever_type,
            **passage_retriever_kwargs
        )
        self._device = device
        self.num_gpus = num_gpus
        # Initialize vLLM model
        self.llm = LLM(model=model_name, tensor_parallel_size=num_gpus)

    def predict_with_llm(self, request_dict, max_new_tokens, prepocess, batch_size=1):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )
        
        # Configure generation
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,  # deterministic
        )
        
        outputs = []
        for i in tqdm(range(0, len(self.prompts), batch_size)):
            batch_prompts = self.prompts[i : i + batch_size]
            results = self.llm.generate(batch_prompts, sampling_params)
            for r in results:
                if not r.outputs:
                    outputs.append("")
                else:
                    outputs.append(r.outputs[0].text.strip())
        return outputs

    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file")
    parser.add_argument("--custom_qrel_path", type=str, help="path to custom qrel file")
    parser.add_argument("--corpus", type=str, help="corpus identifier (for backward compatibility)")
    parser.add_argument("--query_mapping_file", type=str, help="path to query mappings file")
    parser.add_argument("--query_mapping_format", type=str, default="auto",
                        help="format of query mapping file (auto, json, json_simple, tsv, xml)")
    parser.add_argument("--passage_retriever_type", type=str,
                        help="type of passage retriever (pyserini_msmarco_v1, json_file, etc.)")
    parser.add_argument("--passage_file_path", type=str, help="path to passage file")
    parser.add_argument("--index_path", type=str, help="path to custom Pyserini index")
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument(
        "--prompt_type", type=str, help="Prompt type. Supported types: [bing, basic]."
    )
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use")

    args = parser.parse_args()
    load_dotenv()

    # Prepare passage retriever kwargs
    passage_retriever_kwargs = {}
    if args.passage_file_path:
        passage_retriever_kwargs['passage_file_path'] = args.passage_file_path
    if args.index_path:
        passage_retriever_kwargs['index_path'] = args.index_path

    judge = OSLLMJudge(
        qrel=args.qrel,
        model_name=args.model,
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        few_shot_count=args.few_shot_count,
        device=args.device,
        num_gpus=args.num_gpus,
        corpus=args.corpus,
        query_mapping_file=args.query_mapping_file,
        custom_qrel_path=args.custom_qrel_path,
        query_mapping_format=args.query_mapping_format,
        passage_retriever_type=args.passage_retriever_type,
        **passage_retriever_kwargs
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()