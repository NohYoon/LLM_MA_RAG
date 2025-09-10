[OCR 관련]

conda activate dots_ocr
cd dots.ocr
# You need to register model to vllm at first
export hf_model_path=./weights/DotsOCR  # Path to your downloaded model weights, Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`  # If you downloaded model weights by yourself, please replace `DotsOCR` by your model saved directory name, and remember to use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) 

# launch vllm server
CUDA_VISIBLE_DEVICES=3 vllm serve ${hf_model_path} --port 8001 --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code

[OCR API]
uvicorn api:app --host 0.0.0.0 --port 8002 &

[GPT-OSS 20b docker]
conda activate gpt

docker run --gpus '"device=2"' \
    -p 8003:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.10.1 \
    --model openai/gpt-oss-20b

[GPT-OSS 120b transformers]
conda activate gpt
# 모델 가중치를 저장할 로컬 디렉토리 (매번 다운로드 방지)
export MODEL_DIR=$HOME/models

docker run --rm \
  --gpus '"device=0,1,2"' \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
  -e HF_HOME=/data \
  -e HUGGINGFACE_HUB_CACHE=/data \
  --shm-size=1g \
  -p 8080:80 \
  -v $MODEL_DIR:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
    --model-id openai/gpt-oss-120b \
    --sharded true \
    --num-shard 3 \
    --dtype auto \
    --max-input-tokens 8192 \
    --max-total-tokens 65536