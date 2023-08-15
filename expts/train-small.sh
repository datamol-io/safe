accelerate launch --config_file config/accelerate.yaml \
    scripts/model_trainer.py --tokenizer "tokenizer/tokenizer-custom.json" \
    --dataset data/ --text_column "input" \
    --is_tokenized False --streaming True \
    --num_labels 1 --include_descriptors False \
    --gradient_accumulation_steps 2 --wandb_watch 'gradients' \
    --per_device_train_batch_size 32 --num_train_epochs 5 --save_steps 2000 --save_total_limit 10 \
    --eval_accumulation_steps 100 --logging_steps 200 --logging_first_step True \
    --save_safetensors True --do_train True --output_dir output/test/ \
    --learning_rate 5e-4 --warmup_steps 1000 --gradient_checkpointing True --max_steps 15_000
