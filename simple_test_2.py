import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# 加载模型和分词器
model_name = "model/Llama-7b"  # 替换为你自己的模型路径
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 如果有GPU，可以使用CUDA加速
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入文本
input_text = []
for i in range(1024):  # 假设有10个输入文本
    input_text.append(f"Once upon a time, the cat is playing.")

# 编码输入
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 生成设置
max_length = 1024  # 最大生成长度

# 使用generate方法生成文本
generated_ids = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    do_sample=True,       # 如果需要引入采样（如top-k或top-p），可以设置为True
    top_k=50,             # top-k采样，限制选择前k个最可能的词
    top_p=0.95,           # top-p（核采样），控制生成的多样性
    temperature=0.7       # 控制生成的多样性
)

# 解码生成的文本
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Final generated text: {generated_text}")
