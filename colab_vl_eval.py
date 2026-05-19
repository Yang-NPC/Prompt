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
        default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        help="Vision-language model to evaluate.",
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
        (
            "中文_图表数值对齐",
            """你是一名擅长 GRE 数学图表题的数据解读专家。
请只依据图片中的信息作答，重点读取标题、坐标轴、图例、单位、年份、百分比、数值刻度和不同子图之间的关系。
先在心里确定题目问的是哪个图、哪个年份、哪个类别或哪几个量；再根据图中读数进行必要计算；最后把计算结果和所有选项逐一比较。
如果题目问“近似值”或“最接近哪一项”，请选择与计算结果最接近的选项，而不是看起来顺眼的选项。
如果是多选题，请分别判断每个选项是否由图表必然推出，并按字母顺序输出所有正确选项。

课程：{course}
题型：{type}
问题：{question}
选项：{options}

只输出最终选项字母，例如 B 或 ACD。不要解释，不要输出选项内容。""",
        ),
        (
            "中文_逐项排除",
            """请根据图片解答这道 {course} 的 {type}。
在给出答案前，请在心里完成以下步骤：
1. 明确题目要求的对象，是年份、类别、比例、差值、总量、平均值、角度还是多个陈述判断。
2. 从图片中读取相关数值，注意单位、刻度间隔、小数/百分数、以及图例颜色或阴影。
3. 做对应运算：百分比变化、比例比较、求和、求差、平均、扇形角度或真假判断。
4. 把结果与 A/B/C/D/E 每个选项比较，排除不匹配的选项。
5. 多选题时不要只选一个看起来最像的选项，而是逐项判断所有选项。

问题：{question}
选项：{options}

最终回答只能包含选项字母 A-E；多选时直接连写，如 BC 或 ACD。不要输出任何解释。""",
        ),
        (
            "中文_GRE数据分析",
            """这是一道 GRE 数学数据分析题，图片可能包含柱状图、折线图、饼图、箱线图或多个相关图表。
请优先理解图表本身：标题说明了总体对象，坐标轴说明了变量和单位，图例说明了类别，刻度决定了读数精度。
不要根据选项分布猜答案。请先读图，再计算，再选择。
如果两个选项接近，选择最接近计算值的那个。
如果题目要求选择所有正确陈述，只选择从图表中一定为真的陈述。

题型：{type}
问题：{question}
选项：{options}

只返回最终答案字母。单选返回一个字母；多选返回多个字母且按字母顺序排列。""",
        ),
        (
            "中文_先读图后算题",
            """先读图片，再读题目，再选择答案。
请从图片中找到相关数值、类别、年份和比较对象，然后判断题目需要哪种运算：
- 百分比增加或减少：使用（新值 - 旧值）/ 旧值
- 比例比较：比较商，而不是只看差值
- 总数或平均数：合并所有相关可见数值
- 饼图角度：使用 部分 / 总体 * 360
- 多选题：逐条检验每个陈述是否正确

课程：{course}
题型：{type}
问题：{question}
选项：{options}

只输出按字母顺序排列的选项字母，不要解释。""",
        ),
        (
            "中文_多选防漏选",
            """请根据图片和选项解题。
特别注意：如果题型是多选题，正确答案可能包含多个字母。除非确实只有一个陈述正确，否则不要只输出一个字母。
多选题请将每个选项分别与图表数据核对，保留所有正确项，去掉所有错误项。
单选题请根据图表读数和计算结果选择最接近的一个选项。

题型：{type}
问题：{question}
选项：{options}

答案格式只能是字母，例如 A、D、BC、ACD。不要解释。""",
        ),
        (
            "中文_单位刻度检查",
            """请用图片中的图表回答这道 {course} {type}。
选择前请在心里检查：
- 题目对应的是哪一个图或哪一个子图
- 坐标轴刻度和单位是什么
- 图中数值是百分数、美元、人数、票数、角度、百万还是其他单位
- 题目问的是增加、减少、少多少、多多少、最接近、比例相等，还是所有正确陈述
- 选项中的百分比是否用小数表示，例如 0.36 表示 36%

问题：{question}
选项：{options}

请选择最符合图表计算结果的选项。只输出选项字母。""",
        ),
        (
            "中文_隐式逐步推理",
            """请在心里逐步解决这道图表题，但最终不要展示步骤。
你需要先读图，找出题目所需的精确或近似数据；然后完成计算；再与所有选项比较；最后只给出答案。
若为多选题，请在心里逐项检查所有陈述，并输出所有正确选项。

课程：{course}
题型：{type}
问题：{question}
选项：{options}

答案格式：只允许输出 A、B、C、D、E 或类似 BC、ACD 的多字母答案。""",
        ),
        (
            "中文_强约束只输出",
            """你正在解一道英文 GRE 数学图表题。题干和选项可能是英文，但请用图片中的图表数据进行中文思考。
请不要被选项顺序诱导。先确定题目中的关键词含义，例如起止范围、增加或减少、比例、平均值、最接近哪一项、哪些陈述为真。
然后读取图表对应数据并完成计算。
如果答案是百分比，注意选项可能用小数表示百分比。
如果答案是多个陈述，必须输出所有正确字母，不能漏选。

题型：{type}
问题：{question}
选项：{options}

最终只输出答案字母，不要输出中文、英文解释、标点或空格。""",
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

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="auto",
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
