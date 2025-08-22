import argparse
import os
from typing_extensions import Optional

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from retry import retry
from tqdm import tqdm

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class GeminiJudge(LLMJudge):
    def __init__(
        self,
        qrel: str = None,
        model_name: str = None,
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 0,
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
        self.create_gemini_client()

    def create_gemini_client(self):
        vertexai.init(
            project=os.environ["GCLOUD_PROJECT"], location=os.environ["GCLOUD_REGION"]
        )
        self.client = GenerativeModel(self.model_name)

    @retry(tries=3, delay=0.1)
    def run_gemini(self, prompt, max_new_tokens):
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0,
                    top_p=1,
                ),
            )
            output = response.text.lower() if response.text else ""
        except Exception as e:
            print(f"Encountered {e} for {prompt}")
            output = ""
        return output

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
    ):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )

        outputs = [
            self.run_gemini(prompt, max_new_tokens) for prompt in tqdm(self.prompts)
        ]
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

    args = parser.parse_args()
    load_dotenv()

    # Prepare passage retriever kwargs
    passage_retriever_kwargs = {}
    if args.passage_file_path:
        passage_retriever_kwargs['passage_file_path'] = args.passage_file_path
    if args.index_path:
        passage_retriever_kwargs['index_path'] = args.index_path

    judge = GeminiJudge(
        qrel=args.qrel,
        model_name=args.model,
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        few_shot_count=args.few_shot_count,
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