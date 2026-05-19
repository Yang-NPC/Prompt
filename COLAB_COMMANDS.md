# Colab Commands

Use these cells in Colab. The notebook file is intentionally ignored by git, so this tracked file is the source of truth for the current run setup.

## 1. Config

```python
REPO_URL = "https://github.com/Yang-NPC/Prompt.git"
BRANCH = "main"
REPO_DIR = "/content/prompt_test_repo"
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
RESULT_FILE = "colab_qwen72b_chinese_prompt_results.json"
```

## 2. Clone Latest Repo

```python
%cd /content
!rm -rf "$REPO_DIR"
!git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
%cd "$REPO_DIR"
!git log --oneline -n 1
```

Expected latest commit should include the 72B Chinese prompt change.

## 3. Install Dependencies

```python
!python -V
!nvidia-smi
!pip install -U pip
!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -U git+https://github.com/huggingface/transformers accelerate pillow sentencepiece qwen-vl-utils
```

## 4. Run Evaluation

```python
!python colab_vl_eval.py \
  --model-id "$MODEL_ID" \
  --output "$RESULT_FILE"
```

The run should print an effective config with:

```text
search_grid=8
top_k=8
```

The model should be:

```text
Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```

## 5. Inspect Result

```python
import json
from pathlib import Path

result_path = Path(REPO_DIR) / RESULT_FILE
result = json.loads(result_path.read_text(encoding="utf-8"))
print(result["model_id"])
print(result["run_config"])
result["best"]
```

## 6. Push Result

Set a fresh GitHub token first. Do not paste token screenshots into chat.

```python
import os
os.environ["GITHUB_USER"] = "Yang-NPC"
os.environ["GITHUB_TOKEN"] = "YOUR_NEW_TOKEN"
```

```python
!git remote set-url origin https://$GITHUB_USER:$GITHUB_TOKEN@github.com/Yang-NPC/Prompt.git
!git status --short
!git add "$RESULT_FILE"
!git commit -m "Add Qwen 72B Chinese prompt results" || true
!git push origin "$BRANCH"
!git remote set-url origin https://github.com/Yang-NPC/Prompt.git
```
