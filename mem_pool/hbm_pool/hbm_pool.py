import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import threading

@dataclass
class KVCacheBlock:
    """KV Cache内存块"""
    index: int
    sequence_id: int = -1
    past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

class TransformerKVCache:
    def __init__(
        self,
        model_name: str,
        num_blocks: int,
        max_seq_len: int,
        device: str = "cuda"
    ):
        """
        初始化Transformer模型和KV Cache管理器
        
        Args:
            model_name: Hugging Face模型名称
            num_blocks: 内存池中的块数量
            max_seq_len: 最大序列长度
            device: 运行设备
        """
        self.device = device
        self.max_seq_len = max_seq_len
        
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 获取模型配置
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 初始化KV Cache池
        self.blocks = []
        self.free_blocks = set(range(num_blocks))
        self.sequence_blocks: Dict[int, List[int]] = {}
        
        # 创建内存块
        for i in range(num_blocks):
            # 为每一层创建key和value缓存
            past_key_values = tuple(
                (
                    torch.zeros(
                        (1, self.num_heads, max_seq_len, self.head_dim),
                        dtype=torch.float16,
                        device=device
                    ),
                    torch.zeros(
                        (1, self.num_heads, max_seq_len, self.head_dim),
                        dtype=torch.float16,
                        device=device
                    )
                )
                for _ in range(self.num_layers)
            )
            
            self.blocks.append(KVCacheBlock(
                index=i,
                past_key_values=past_key_values
            ))
        
        self.lock = threading.Lock()

    def allocate(self, sequence_id: int) -> Optional[int]:
        """分配一个KV Cache块"""
        with self.lock:
            if not self.free_blocks:
                return None
            
            block_idx = self.free_blocks.pop()
            block = self.blocks[block_idx]
            block.sequence_id = sequence_id
            
            if sequence_id not in self.sequence_blocks:
                self.sequence_blocks[sequence_id] = []
            self.sequence_blocks[sequence_id].append(block_idx)
            
            return block_idx

    def free(self, sequence_id: int) -> None:
        """释放序列的KV Cache"""
        with self.lock:
            if sequence_id not in self.sequence_blocks:
                return
            
            for block_idx in self.sequence_blocks[sequence_id]:
                block = self.blocks[block_idx]
                block.sequence_id = -1
                # 清零缓存
                for layer_cache in block.past_key_values:
                    for tensor in layer_cache:
                        tensor.zero_()
                self.free_blocks.add(block_idx)
            
            del self.sequence_blocks[sequence_id]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        使用KV Cache生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: 核采样参数
        
        Returns:
            生成的文本，如果分配失败返回None
        """
        # 分配新的sequence_id和KV Cache
        sequence_id = hash(prompt) % 10000000  # 简单的序列ID生成
        block_idx = self.allocate(sequence_id)
        
        if block_idx is None:
            return None
        
        try:
            # 编码输入
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            # 初始化past_key_values
            past_key_values = None
            current_length = 0
            
            generated_tokens = []
            
            for _ in range(max_new_tokens):
                # 准备模型输入
                model_inputs = {
                    "input_ids": input_ids[:, -1:] if past_key_values is not None else input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                
                # 模型推理
                outputs = self.model(**model_inputs)
                
                # 更新KV Cache
                past_key_values = outputs.past_key_values
                
                # 采样下一个token
                next_token_logits = outputs.logits[:, -1, :] / temperature
                filtered_logits = self._top_p_filtering(next_token_logits, top_p)
                next_token = torch.multinomial(
                    torch.softmax(filtered_logits, dim=-1),
                    num_samples=1,
                )
                
                generated_tokens.append(next_token.item())
                
                # 更新input_ids和attention_mask
                input_ids = next_token.unsqueeze(0)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device)
                ], dim=-1)
                
                current_length += 1
                
                # 检查是否生成了终止token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
            # 解码生成的文本
            generated_text = self.tokenizer.decode(generated_tokens)
            return generated_text
            
        finally:
            # 释放KV Cache
            self.free(sequence_id)

    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """实现top-p (nucleus) 采样过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1
        )
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits

    def get_stats(self) -> Dict:
        """获取KV Cache使用统计"""
        with self.lock:
            return {
                "total_blocks": len(self.blocks),
                "free_blocks": len(self.free_blocks),
                "used_blocks": len(self.blocks) - len(self.free_blocks),
                "active_sequences": len(self.sequence_blocks),
                "max_seq_len": self.max_seq_len,
                "memory_per_block": (
                    self.num_layers * 2 * # 2 for key and value
                    self.num_heads *
                    self.max_seq_len *
                    self.head_dim *
                    2  # 2 bytes for float16
                )
            }

# 使用示例
def main():
    # 初始化KV Cache管理器
    cache_manager = TransformerKVCache(
        model_name="meta-llama/Llama-2-7b-hf",  # 或其他支持的模型
        num_blocks=10,  # 内存池中的块数量
        max_seq_len=2048,  # 最大序列长度
        device="cuda"
    )
    
    # 生成文本
    prompt = "Once upon a time"
    generated_text = cache_manager.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
    
    if generated_text is not None:
        print(f"Generated text: {generated_text}")
    else:
        print("Failed to allocate KV Cache")
    
    # 打印内存使用统计
    print(f"Cache stats: {cache_manager.get_stats()}")

if __name__ == "__main__":
    main()