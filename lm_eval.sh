uv run lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,max_model_len=2048 \
    --tasks hellaswag,kobest_hellaswag \
    --batch_size auto \
    --limit 100 \
    --device mps \
    --verbosity DEBUG