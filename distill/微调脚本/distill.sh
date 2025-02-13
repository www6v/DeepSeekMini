
# step1 
### 数据清洗, 详细看文档, 关注下数据的格式


# step2
### 在LLama-Factory中注册自定义数据集，找到dataset_info.json

```

"Distill": {
    "file_name": "/root/autodl-tmp/DeepSeek-R1-Distill/dataset/Distill_data_17ktrain.arrow",
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


# step3
### 使用Llama-Factory full脚本进行模型蒸馏，即进行全量指令微调

$ cd /root/autodl-tmp/DeepSeek-R1-Distill/LLaMA-Factory
$ FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train
examples/train_full/qwen2_full_sft.yaml