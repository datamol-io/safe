accelerate launch --config_file config/accelerate.yaml \
    scripts/model_trainer.py --tokenizer "tokenizer/tokenizer-custom.json" \
    --dataset ~/data/ --text_column "input" \
    --is_tokenized False --streaming True \
    --num_labels 1 --include_descriptors False \
    --gradient_accumulation_steps 2 --wandb_watch 'gradients' \
    --per_device_train_batch_size 64 --num_train_epochs 2 --save_steps 5000 --save_total_limit 10 \
    --eval_accumulation_steps 100 --logging_steps 500 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir output/safe/ \
    --learning_rate 5e-5 --warmup_steps 2500 --gradient_checkpointing True --max_steps 30_000_000