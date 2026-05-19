#!/usr/bin/env python3

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "英文数学图表" / "data.json"
DEFAULT_IMAGE_DIR = ROOT / "英文数学图表" / "images"
DEFAULT_TEMPLATE = ROOT / "template.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Vision-language model, recommended within 1B-4B.",
    )
    parser.add_argument("--dataset-json", default=str(DEFAULT_DATASET))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--template-file", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output", default="colab_eval_results.json")
    parser.add_argument("--stage1-ids", default="1,3,5,6,8,9,11,12,15,18")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_template(template_file):
    raw = Path(template_file).read_text(encoding="utf-8")
    match = re.search(r'prompt_template:\s*"(.*)"\s*top_p:', raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse prompt template from {template_file}")
    return bytes(match.group(1), "utf-8").decode("unicode_escape")


def build_prompt_variants(base_template):
    return [
        ("template_baseline", base_template),
        (
            "strict_letter_only",
            """Answer the question using the image.
You are an expert in {course}.
Question type: {type}
Question: {question}
Options: {options}
Return only the correct option letter or letters in alphabetical order, such as A or ACD.
Do not return words, punctuation, explanations, or option text.""",
        ),
        (
            "math_first_concise",
            """Use the image and the question to determine the correct answer.
This is a {type} in {course}.
Question: {question}
Options: {options}
Think carefully about the quantities shown in the image, then output only the option letter.
If multiple answers are correct, output the letters in alphabetical order with no spaces.""",
        ),
        (
            "chart_compare_all_options",
            """You are solving a {type} question in {course} from a chart or graph.
Read the image carefully, estimate the relevant values, and compare every option before choosing.
Question: {question}
Options: {options}
Rules:
1. Use the image, not world knowledge.
2. If the values are approximate, choose the closest option after comparing all choices.
3. For multi-select questions, test each option independently and return all correct letters in alphabetical order with no separators.
4. Output only the final option letter or letters.""",
        ),
        (
            "extract_then_answer",
            """Solve this {type} question in {course} using the image.
First identify the key numbers, ratios, or trends shown in the chart mentally.
Then answer the question by matching the best option.
Question: {question}
Options: {options}
Return only the option letter.
If more than one option is correct, return the letters in alphabetical order with no spaces.""",
        ),
    ]


def build_search_grid(prompt_variants):
    top_ps = [0.2, 0.5, 0.9, 1.0]
    temperatures = [0.0, 0.05, 0.1]
    grid = []
    for prompt_name, prompt_template in prompt_variants:
        for top_p in top_ps:
            for temperature in temperatures:
                grid.append(
                    {
                        "prompt_name": prompt_name,
                        "prompt_template": prompt_template,
                        "top_p": top_p,
                        "temperature": temperature,
                    }
                )
    return grid


def load_dataset(dataset_json, limit):
    rows = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    if limit > 0:
        return rows[:limit]
    return rows


def parse_ids(raw_ids):
    return {int(x.strip()) for x in raw_ids.split(",") if x.strip()}


def format_options(options):
    return "; ".join(f"{key}: {options[key]}" for key in sorted(options))


def normalize_answer(text):
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.upper().strip()
    patterns = [
        r"ANSWER\s*[:=]\s*([A-E](?:\s*[,/ ]\s*[A-E])*)",
        r"OPTION\s*[:=]\s*([A-E](?:\s*[,/ ]\s*[A-E])*)",
        r"\b([A-E](?:\s*[,/ ]\s*[A-E]){0,4})\b",
    ]
    candidate = ""
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            candidate = matches[-1]
            break
    if not candidate:
        candidate = "".join(re.findall(r"[A-E]", cleaned)[-4:])
    deduped = []
    for letter in re.findall(r"[A-E]", candidate):
        if letter not in deduped:
            deduped.append(letter)
    return "".join(sorted(deduped))


def select_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_id):
    dtype = select_dtype()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_inference(model, processor, image_path, prompt, top_p, temperature, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = move_to_device(inputs, model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate_setting(model, processor, dataset, image_dir, setting, max_new_tokens):
    correct = 0
    latencies = []
    details = []

    for item in dataset:
        prompt = setting["prompt_template"].format(
            course=item["course"],
            type=item["type"],
            question=item["question"],
            options=format_options(item["options"]),
        )
        image_path = Path(image_dir) / item["image"]
        started = time.time()
        raw = run_inference(
            model,
            processor,
            image_path,
            prompt,
            setting["top_p"],
            setting["temperature"],
            max_new_tokens,
        )
        elapsed = time.time() - started
        pred = normalize_answer(raw)
        gold = normalize_answer(item["answer"])
        ok = pred == gold
        if ok:
            correct += 1
        latencies.append(elapsed)
        details.append(
            {
                "id": item["id"],
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "correct": ok,
                "latency_sec": round(elapsed, 2),
            }
        )
        print(
            f"  q{item['id']:02d}: pred={pred or '-'} gold={gold} {'OK' if ok else 'MISS'} ({elapsed:.1f}s)",
            flush=True,
        )

    return {
        "prompt_name": setting["prompt_name"],
        "prompt_template": setting["prompt_template"],
        "top_p": setting["top_p"],
        "temperature": setting["temperature"],
        "correct": correct,
        "total": len(dataset),
        "accuracy": correct / len(dataset),
        "avg_latency_sec": round(statistics.mean(latencies), 2),
        "details": details,
    }


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_json, args.limit)
    stage1_ids = parse_ids(args.stage1_ids)
    stage1_dataset = [item for item in dataset if item["id"] in stage1_ids]
    base_template = load_template(args.template_file)
    prompt_variants = build_prompt_variants(base_template)
    search_grid = build_search_grid(prompt_variants)

    print(f"Loading model: {args.model_id}", flush=True)
    model, processor = load_model(args.model_id)

    stage1_results = []
    print(f"Stage 1 on {len(stage1_dataset)} representative questions", flush=True)
    for index, setting in enumerate(search_grid, start=1):
        print(
            f"[{index}/{len(search_grid)}] prompt={setting['prompt_name']} top_p={setting['top_p']} temperature={setting['temperature']}",
            flush=True,
        )
        result = evaluate_setting(
            model,
            processor,
            stage1_dataset,
            args.image_dir,
            setting,
            args.max_new_tokens,
        )
        stage1_results.append(result)
        print(
            f"  -> stage1 accuracy={result['correct']}/{result['total']} ({result['accuracy']:.3f})",
            flush=True,
        )

    stage1_results.sort(key=lambda x: (x["accuracy"], -x["avg_latency_sec"]), reverse=True)
    top_candidates = stage1_results[: args.top_k]

    full_results = []
    print(f"\nStage 2 on full dataset ({len(dataset)} questions)", flush=True)
    for index, candidate in enumerate(top_candidates, start=1):
        print(
            f"[{index}/{len(top_candidates)}] prompt={candidate['prompt_name']} top_p={candidate['top_p']} temperature={candidate['temperature']}",
            flush=True,
        )
        setting = {
            "prompt_name": candidate["prompt_name"],
            "prompt_template": candidate["prompt_template"],
            "top_p": candidate["top_p"],
            "temperature": candidate["temperature"],
        }
        result = evaluate_setting(
            model,
            processor,
            dataset,
            args.image_dir,
            setting,
            args.max_new_tokens,
        )
        full_results.append(result)
        print(
            f"  -> full accuracy={result['correct']}/{result['total']} ({result['accuracy']:.3f})",
            flush=True,
        )

    full_results.sort(key=lambda x: (x["accuracy"], -x["avg_latency_sec"]), reverse=True)
    output = {
        "model_id": args.model_id,
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "image_dir": str(Path(args.image_dir).resolve()),
        "stage1_results": stage1_results,
        "full_results": full_results,
        "best": full_results[0] if full_results else None,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}", flush=True)
    if output["best"]:
        print(json.dumps(output["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
