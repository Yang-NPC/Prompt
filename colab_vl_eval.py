#!/usr/bin/env python3

import argparse
import json
import re
import statistics
import time
from pathlib import Path

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
    parser.add_argument("--top-k", type=int, default=8)
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
            "chart_math_compare",
            """You are an expert at GRE quantitative reasoning with charts.
Use the image as the source of data. Read labels, axes, legends, units, and percentages carefully.
Solve the question, then compare your computed or estimated result with every option.
For approximate questions, choose the closest option, not simply the first plausible option.
For multi-select questions, judge each option independently and return every correct letter in alphabetical order.

Question type: {type}
Question: {question}
Options: {options}

Output only the option letter or letters, such as B or ACD. Do not output explanations.""",
        ),
        (
            "option_elimination",
            """Answer this {course} {type} using the chart in the image.
Mentally do these steps before answering:
1. Identify exactly which graph, year, category, or group the question asks about.
2. Extract the needed values from the image, including units and approximate values.
3. Perform the required arithmetic: percent change, ratio, total, difference, average, or angle.
4. Check all options against the result and eliminate options that do not match.
5. If the question asks for all true statements, evaluate A, B, C, and D separately.

Question: {question}
Options: {options}

Final response must be only letters A-E, with no spaces, punctuation, or explanation.""",
        ),
        (
            "gre_chart_solver",
            """You are solving a GRE math data interpretation problem.
The image may contain bar graphs, pie charts, line charts, boxplots, or multiple related charts.
Pay attention to titles, legends, axes, scales, years, and whether numbers are percentages or absolute amounts.
Do not guess from option patterns. First infer the relevant chart values, then calculate.
When two options are close, prefer the one closest to the calculated value.
For multi-answer questions, select all and only the statements that must be true from the chart.

Question: {question}
Options: {options}
Question type: {type}

Return only the final answer letter or letters.""",
        ),
        (
            "visual_data_first",
            """Focus on the image first, then the question.
Use the visual data to identify the relevant values, categories, and comparisons.
After reading the question, decide what operation is needed:
- percent increase/decrease: use (new - old) / old
- ratio comparison: compare quotients, not raw differences
- total or average: combine all relevant visible values
- pie chart angle: use part / whole * 360
- multi-select: test every statement separately

Course: {course}
Type: {type}
Question: {question}
Options: {options}

Only output the option letter or letters in alphabetical order.""",
        ),
        (
            "multi_select_guard",
            """Solve the problem from the image and options.
Important: if this is a multi-select question, the correct answer may contain more than one letter.
Do not choose a single letter for a multi-select question unless exactly one statement is true.
Evaluate each option independently against the chart data.
For single-choice questions, calculate the requested value and choose the closest matching option.

Question type: {type}
Question: {question}
Options: {options}

Return only the final letter string, for example A, D, BC, or ACD.""",
        ),
        (
            "chart_units_and_scale",
            """Use the chart image to answer the {course} {type}.
Before choosing, mentally verify:
- the chart title and which sub-chart applies
- axis scale and units
- whether values are percents, dollars, counts, angles, or millions
- whether the question asks for increase, decrease, less than, greater than, closest to, or all true statements
- whether option values are written as decimals for percentages

Question: {question}
Options: {options}

Select the option that best matches the image-based calculation.
Output only the option letter or letters.""",
        ),
        (
            "structured_private_reasoning",
            """Privately solve this chart problem step by step, but do not show the steps.
Read the image, identify the exact relevant data, calculate the requested quantity, compare with all options, and then provide only the answer.
If the problem is multi-select, privately check each statement and output all true letters in alphabetical order.

Course: {course}
Question type: {type}
Question: {question}
Options: {options}

Answer format: only A, B, C, D, E, or a multi-letter string like BC or ACD.""",
        ),
    ]


def build_search_grid(prompt_variants):
    top_ps = [0.5]
    temperatures = [0.05]
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
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_id):
    from transformers import AutoModelForImageTextToText, AutoProcessor

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
    import torch
    from PIL import Image

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
    stage1_dataset_ids = [item["id"] for item in stage1_dataset]

    print(f"Loading model: {args.model_id}", flush=True)
    print(
        "Effective config: "
        f"stage1_ids={sorted(stage1_ids)}, "
        f"stage1_dataset_ids={stage1_dataset_ids}, "
        f"top_k={args.top_k}, "
        f"max_new_tokens={args.max_new_tokens}, "
        f"search_grid={len(search_grid)}",
        flush=True,
    )
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
        "run_config": {
            "stage1_ids_arg": args.stage1_ids,
            "stage1_ids": sorted(stage1_ids),
            "stage1_dataset_ids": stage1_dataset_ids,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "search_grid_count": len(search_grid),
            "prompt_variant_names": [name for name, _ in prompt_variants],
        },
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
