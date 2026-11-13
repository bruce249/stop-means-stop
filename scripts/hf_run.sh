#!/usr/bin/env bash
# Runs inside a Hugging Face Job. The repo is mounted read-only at /code.
# One GPU, sequential. Training hyperparameters match the Kaggle notebook.
set -euo pipefail

echo "==== gpu ===="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cp -a /code /tmp/sms
cd /tmp/sms

uv venv --python 3.12 .venv -q
ok=0
for idx in cu128 cu126; do
  if uv pip install -q --python .venv "torch==2.11.0" torchvision --index-url "https://download.pytorch.org/whl/${idx}"; then
    ok=1
    echo "torch from ${idx}"
    break
  fi
done
if [ "$ok" != "1" ]; then
  uv pip install -q --python .venv "torch==2.11.0" torchvision
fi
uv pip install -q --python .venv -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print('gpu matmul ok', tuple((x@x).shape))"
.venv/bin/python -c "import agenttune, auditkit, curatorkit, trl, transformers, peft; print('agenttune', agenttune.__version__, '| trl', trl.__version__)"

PY=".venv/bin/python"
mkdir -p runs

SMOKE="${SMOKE:-0}"
RUN_CONTROL="${RUN_CONTROL:-1}"
SECOND_SEED="${SECOND_SEED:-1}"
PUSH="${PUSH:-0}"
EVAL_BATCH="${EVAL_BATCH:-16}"

if [ "$SMOKE" = "1" ]; then
  echo "==== smoke ===="
  "$PY" scripts/make_data.py --out data --scale 0.1
  LIMIT=(--limit 24)
  TRAIN=(--max-examples 60)
  SECOND_SEED=0
  PUSH=0
else
  echo "==== full run ===="
  "$PY" scripts/make_data.py --out data
  LIMIT=()
  TRAIN=()
fi

"$PY" scripts/evaluate.py --policy oracle --name oracle "${LIMIT[@]}"
"$PY" scripts/evaluate.py --policy never_stop --name never_stop "${LIMIT[@]}"
"$PY" scripts/evaluate.py --policy hf --name base --batch-size "$EVAL_BATCH" "${LIMIT[@]}"
"$PY" scripts/train.py --split main --out runs/adapter-main "${TRAIN[@]}"
"$PY" scripts/evaluate.py --policy hf --name tuned --adapter runs/adapter-main --batch-size "$EVAL_BATCH" "${LIMIT[@]}"

if [ "$RUN_CONTROL" = "1" ]; then
  "$PY" scripts/train.py --split control_no_interrupts --out runs/adapter-control "${TRAIN[@]}"
  "$PY" scripts/evaluate.py --policy hf --name control --adapter runs/adapter-control --batch-size "$EVAL_BATCH" "${LIMIT[@]}"
fi

if [ "$SECOND_SEED" = "1" ]; then
  "$PY" scripts/train.py --split main --seed 7 --out runs/adapter-s2 "${TRAIN[@]}"
  "$PY" scripts/evaluate.py --policy hf --name tuned_s2 --adapter runs/adapter-s2 --batch-size "$EVAL_BATCH" "${LIMIT[@]}"
fi

"$PY" scripts/report.py --runs runs
echo "==== results.md ===="
cat runs/results.md

if [ "$PUSH" = "1" ]; then
  "$PY" scripts/push_to_hub.py --user "$HF_USER" --github "${GITHUB:-https://github.com/heisenbergk146/stop-means-stop}"
fi
