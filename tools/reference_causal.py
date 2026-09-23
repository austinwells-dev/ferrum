"""Token-ID matched local Transformers reference for Phase 6 decoder models."""

import argparse
import json
import time

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Local official safetensors checkpoint directory")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="JSON array of input token IDs")
    prompt.add_argument("--prompt-text", help="Raw text to tokenize without a chat template")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat raw prompt text")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--full-context", action="store_true", help="Recompute the whole prefix each step"
    )
    args = parser.parse_args()
    if args.prompt_ids:
        ids = json.loads(args.prompt_ids)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        ids = tokenizer(args.prompt_text * args.repeat, add_special_tokens=False)["input_ids"]
    if not ids or args.steps < 1 or args.repeat < 1:
        parser.error("prompt IDs must be nonempty and steps positive")

    torch.set_num_threads(4)
    start = time.perf_counter()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", local_files_only=True, use_safetensors=True
    ).eval()
    load_s = time.perf_counter() - start
    generated = []
    rows = []
    cache = None
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        for step in range(args.steps):
            start = time.perf_counter()
            output = model(
                input_ids=input_ids,
                past_key_values=None if args.full_context else cache,
                use_cache=not args.full_context,
            )
            if not args.full_context:
                cache = output.past_key_values
            logits = output.logits[0, -1].float()
            token = int(torch.argmax(logits))
            top_values, top_ids = torch.topk(logits, 10)
            rows.append(
                {
                    "step": step,
                    "token": token,
                    "top10": [[int(i), float(v)] for i, v in zip(top_ids, top_values)],
                    "logit_l2": float(torch.linalg.vector_norm(logits)),
                    "elapsed_s": time.perf_counter() - start,
                }
            )
            generated.append(token)
            if args.full_context:
                input_ids = torch.cat((input_ids, torch.tensor([[token]], dtype=torch.long)), dim=1)
            else:
                input_ids = torch.tensor([[token]], dtype=torch.long)
    print(
        json.dumps(
            {
                "model": args.model,
                "model_type": model.config.model_type,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "prompt_ids": ids,
                "prompt_text": args.prompt_text,
                "repeat": args.repeat,
                "full_context": args.full_context,
                "generated_ids": generated,
                "load_s": load_s,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
