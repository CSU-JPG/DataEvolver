set -e
nvidia-smi

eval "$(conda shell.bash hook)"
conda create -n DataEvolver python=3.9 -y
conda activate DataEvolver

python -m pip install --upgrade pip
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
python -m pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -r requirements.txt || true
curl -fsSL https://ollama.com/install.sh | sh

python -m pip install \
    nvidia-cublas-cu11==11.11.3.6 \
    nvidia-cuda-cupti-cu11==11.8.87 \
    nvidia-cuda-nvrtc-cu11==11.8.89 \
    nvidia-cuda-runtime-cu11==11.8.89 \
    nvidia-cudnn-cu11==8.7.0.84 \
    nvidia-cufft-cu11==10.9.0.58 \
    nvidia-curand-cu11==10.3.0.86 \
    nvidia-cusolver-cu11==11.4.1.48 \
    nvidia-cusparse-cu11==11.7.5.86 \
    nvidia-nccl-cu11==2.19.3 \
    nvidia-nvtx-cu11==11.8.86

GPUS=(0 1 2 3 4 5 6 7)
BASE_PORT=11437

for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    PORT=$((BASE_PORT + i))
    LOG_FILE="ollama_gpu${GPU_ID}_port${PORT}.log"
    echo "Starting Ollama on GPU $GPU_ID Port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    OLLAMA_HOST="127.0.0.1:$PORT" \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > "$LOG_FILE" 2>&1 &
    sleep 3
done

sleep 20

ss -tuln | grep 114
nvidia-smi
OLLAMA_HOST=127.0.0.1:11437 ollama pull qwen3-vl:latest
OLLAMA_HOST=127.0.0.1:11437 ollama pull qwen3.5:4b
OLLAMA_HOST=127.0.0.1:11437 ollama pull mistral:latest
OLLAMA_HOST=127.0.0.1:11437 ollama list

pip freeze
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py --config config.yaml --shard-id host-a --keep-rejects