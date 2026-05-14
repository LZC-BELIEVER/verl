from compressed_tensors.offload import dispatch_model
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from llmcompressor import oneshot

# 模型路径
MODEL_ID = "/lanzichang/models/Qwen2.5-Math-7B"
SAVE_DIR = "/lanzichang/models/Qwen2.5-Math-7B-FP8-KV-calibrated"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 校准参数
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# 从 MATH 训练集加载校准数据
# prompt 字段结构：numpy array of dicts with 'role'/'content' keys（对话格式）
df = pd.read_parquet("/lanzichang/data/math/train.parquet")
samples = df["prompt"].tolist()[:NUM_CALIBRATION_SAMPLES]

# 用 apply_chat_template 转成文本
texts = [
    tokenizer.apply_chat_template(
        list(messages),  # numpy array → list
        tokenize=False,
        add_generation_prompt=False,
    )
    for messages in samples
]

# 构建 HuggingFace Dataset
raw_ds = Dataset.from_dict({"text": texts})

def tokenize(example):
    return tokenizer(
        example["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = raw_ds.map(tokenize, remove_columns=raw_ds.column_names)

# 量化 recipe：weight FP8（static）+ activation FP8（static）+ KV cache FP8（static）
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

# 执行一次性校准
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 验证输出
print("\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("How many vertical asymptotes does y=2/(x^2+x-6) have?", return_tensors="pt")
sample = {k: v.to(model.device) for k, v in sample.items()}
output = model.generate(**sample, max_new_tokens=200)
print(tokenizer.decode(output[0]))
print("==========================================\n")

# 保存
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved calibrated model to {SAVE_DIR}")
