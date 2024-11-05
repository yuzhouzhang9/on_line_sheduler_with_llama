import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
from my_log import Logger
# 加载模型和分词器
model_name = "model/Llama-7b"  # 替换为你自己的模型路径
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # 启用 mixed precision
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# 设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token
# 如果有GPU，可以使用CUDA加速
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入文本批次
input_text = [f"Once upon a time" for _ in range(1024)]  # 假设有10个输入文本

# 使用 batch_encode_plus 编码输入，返回批次的张量
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
print(f"Input IDs shape: {input_ids.shape}")  # 输出批次大小和序列长度
# time.sleep(20)

# 生成设置
max_length = 256  # 最大生成长度
generated_ids = input_ids  # 初始生成的 token 序列
past_key_values = None  # 初始化 past_key_values

for step in range(max_length):
    # 使用当前的 generated_ids 和 past_key_values 来生成下一个 token
    with torch.no_grad():  # 关闭梯度以节省显存
        outputs = model(input_ids=generated_ids[:, -1:], past_key_values=past_key_values)
    
    # 获取新的 KV Cache
    past_key_values = outputs.past_key_values

    # 预测的下一个 token
    next_token_logits = outputs.logits[:, -1, :]
    print(f"Step {step}: {next_token_logits.shape}")
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    # 将预测的 token 添加到生成的 ID 序列中
    generated_ids = torch.cat((generated_ids, next_token), dim=1)

    # 清理缓存以释放内存
    torch.cuda.empty_cache()

    # 打印当前生成的文本（可选）
    if step % 10 == 0 or step == max_length - 1:
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"Step {step}: {generated_texts}")

    # 终止条件（例如遇到 EOS token）
    if torch.any(next_token == tokenizer.eos_token_id):
        break

# 最终生成的文本
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Final generated texts: {generated_texts}")
