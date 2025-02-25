
# DataSet 
### 数据清洗, 详细看文档, 关注下数据的格式

```
Distill_data_17k-train.arrow
```

# DataSet 配置
### 在LLama-Factory中注册自定义数据集，找到dataset_info.json

```

"Distill": {
    "file_name": "/data/jupyterfile/wei/distill/Distill_data_17k-train.arrow",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system"
    },
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
}

```

# Train 脚本
###【配置 qwen2_full_sft.yaml】

```

### model
model_name_or_path: /data/jupyterfile/wei/distill/Qwen2.5-1.5B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_offload_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: Distill
template: qwen
cutoff_len: 8192
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./saves/MiniDeepSeekR1/full/original
logging_steps: 1
save_steps: 100
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 12
learning_rate: 1.0e-5
num_train_epochs: 1.0 #3.0 
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

report_to: wandb
run_name: Distill


```


# Train
### 使用Llama-Factory full脚本进行模型蒸馏，即进行全量指令微调

$ cd /data/jupyterfile/wei/distill/LLaMA-Factory-main
$ FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train  examples/train_full/qwen2_full_sft.yaml
