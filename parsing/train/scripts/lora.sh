export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-$((28500 + $RANDOM % 2000))}

output_dir=checkpoint/monkeyocrv2_lora

export MODELSCOPE_CACHE='checkpoint/cache'

get_latest_checkpoint() { 
    local version_folder="$1" 
    resume_from=""
    if [ -z "$version_folder" ]; then 
        echo "Error: no version folder provided to get_latest_checkpoint" 
        return 1 
    fi 
    echo "Checking $version_folder"
    local checkpoints=() 
    while IFS= read -r -d '' dir; do
        checkpoints+=("$dir")
    done < <(find "$version_folder" -type d -name "checkpoint-[0-9]*" -print0 2>/dev/null)

    if [ ${#checkpoints[@]} -eq 0 ]; then
            echo "No checkpoint found in $version_folder"        
            return 2    
    fi
    local latest_num=0
    for cp in "${checkpoints[@]}"; do
        local num
        num=$(echo "$cp" | grep -oE "checkpoint-[0-9]+" | cut -d '-' -f 2)
        if [ -n "$num" ] && [ "$num" -gt "$latest_num" ]; then
            latest_num=$num
            resume_from="$version_folder/checkpoint-$latest_num"
        fi
    done

    echo "Found latest checkpoint: $resume_from"
}

echo "================ check resume: $output_dir \ resume_from_checkpoint: $resume_from_checkpoint"
RESUME_CHECKPOINT_PATH="None"
if [ -d "$output_dir" ]; then
    echo "================ check resume"
    get_latest_checkpoint "$output_dir"
    RESUME_CHECKPOINT_PATH=$resume_from
    echo "resume checkpoint path: $RESUME_CHECKPOINT_PATH"
    if [ -d "$RESUME_CHECKPOINT_PATH" ]; then
        resume_options=" \
                --resume_from_checkpoint $RESUME_CHECKPOINT_PATH \
                "
    else
        if [ -d "$resume_from_checkpoint" ]; then
            resume_options=" \
                    --resume_from_checkpoint $resume_from_checkpoint \
                    "
            echo "using resume checkpoint param path[$resume_from_checkpoint]"
        fi
    fi
else
    if [ -d "$resume_from_checkpoint" ]; then
        resume_options=" \
                --resume_from_checkpoint $resume_from_checkpoint \
                "
        echo "using resume checkpoint param path[$resume_from_checkpoint]"
    else
        echo "save model dir[$output_dir] is not exist"
        resume_options=""
    fi
fi

echo "=== resume_options: $resume_options"

export MAX_PIXELS=1003520

LOG_DIR="lora/train_$(date +%Y%m%d_%H)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/node${NODE_RANK}.log"

swift sft \
    --model zenosai/MonkeyOCRv2-B-Parsing \
    --model_type monkeyocrv2 \
    --template monkeyocrv2 \
    --attn_impl flash_attention_2 \
    --dataset /path/to/train.jsonl \
    --load_from_cache_file True \
    --split_dataset_ratio 0.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle True \
    --streaming False \
    --max_length 16384 \
    --truncation_strategy 'right' \
    --max_pixels $MAX_PIXELS \
    --packing False \
    --padding_free True \
    --train_type 'lora' \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules 'all-linear' \
    --freeze_vit True \
    --freeze_aligner False \
    --freeze_llm False \
    --torch_dtype 'bfloat16' \
    --deepspeed 'zero1' \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type 'cosine' \
    --eval_steps 400000 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --no_add_version \
    --output_dir $output_dir \
    $resume_options 2>&1 | tee -a "$LOG_FILE"
