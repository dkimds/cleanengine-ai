uv run lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen3-0.6B,max_model_len=2048 \
    --tasks hellaswag \
    --batch_size auto \
    --limit 100 \
    --device cpu \
    --verbosity DEBUG