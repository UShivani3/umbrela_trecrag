import argparse
from typing_extensions import Optional
import os

from dotenv import load_dotenv
import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class HGFLLMJudge(LLMJudge):
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
        if self._device == "cuda":
            assert torch.cuda.is_available()
        self.num_gpus = num_gpus

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
        do_sample: bool = True,
        top_p: float = 1.0,
        num_beams: int = 1,
        batch_size: int = 1,
        num_workers: int = 16,
    ):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=os.environ["HF_TOKEN"],
            cache_dir=os.environ["HF_CACHE_DIR"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=os.environ["HF_TOKEN"],
            cache_dir=os.environ["HF_CACHE_DIR"],
        )
        tokenizer.use_default_system_prompt = False
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        model.eval()

        dataset = datasets.Dataset.from_list([{"text": (t)} for t in self.prompts])

        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        )

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        test_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=DataCollatorWithPadding(tokenizer, padding="longest"),
        )

        outputs = []
        for batch in tqdm(test_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(self._device)

            batch_size, seq_length = batch["input_ids"].shape

            with torch.no_grad():
                output = model.generate(
                    **batch,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    num_beams=num_beams,
                )

            for b in range(batch_size):
                if model.config.is_encoder_decoder:
                    output_ids = output[b]
                else:
                    output_ids = output[b, seq_length:]

                outputs.append(
                    tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                )

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
    parser.add_argument("--device", type=str, help="device")

    args = parser.parse_args()
    load_dotenv()

    # Prepare passage retriever kwargs
    passage_retriever_kwargs = {}
    if args.passage_file_path:
        passage_retriever_kwargs['passage_file_path'] = args.passage_file_path
    if args.index_path:
        passage_retriever_kwargs['index_path'] = args.index_path

    judge = HGFLLMJudge(
        qrel=args.qrel,
        model_name=args.model,
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        few_shot_count=args.few_shot_count,
        device=args.device,
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