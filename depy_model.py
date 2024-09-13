# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import ipdb
from llama import Llama
from typing import List
import time
from logger import *
from seq_manager import *
from llama import *
import torch
def main(
    ckpt_dir: str = "llama-2-7b-chat/",
    tokenizer_path: str  = "tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    nproc_per_node:int = 1,
):
    
    # ipdb.set_trace()
    # 开始模型生成
    Logger.set_log_file_path(f"test_time{time.time()}")
    start_load_model_time = time.time()
    Logger.log("开始加载模型")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    # end_time = time.time()
    Logger.log(f"加载模型花费{time.time() - start_load_model_time}")
   
    # ipdb.set_trace() 
    # 查看剩余内存的容量
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory #B
   
    # 1 * 32 * 32 * 128 * 2  512kb
    reserved_length = free_memory // 32 // 32 // 128 // 8

    Logger.log(f"now,free_memory{free_memory//(1024**2)}")
    
    # 预留内存用于KV Cache的存储和分配
    KVCacheManager.initialize_cache(1,reserved_length,32,32,128)
    
    allocated_memory = torch.cuda.memory_allocated(0)
    used_free_memory = total_memory - allocated_memory #B
    # ipdb.set_trace() 
    Logger.log(f"now,free_memory{free_memory//(1024**2)},used{(free_memory - used_free_memory)//(1024**2)},reserve length{reserved_length}")
    # ipdb.set_trace() 
    # ipdb.set_trace()
    # 
    prompts: List[str] = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:Hi everyone,I just """,
        """Translate English to French:sea otter => loutre de merpeppermint => menthe poivréeplush girafe => girafe peluchecheese =>""",
    ]

    # ipdb.set_trace() 
    Logger.log("创建seqEngine")
    #初始化SeqEngine，添加tokeninzer
    SeqEngine.initialize(generator.tokenizer)
    Logger.log(f"add tokenizer success")

    # ipdb.set_trace() 
    # 添加请求
    for request in prompts:
        SeqEngine.add_request([request])
    Logger.log(f"SeqEngine{SeqEngine}")
    Logger.log(f"tokens:{SeqEngine.seq_list}")
    # ipdb.set_trace() 
    
    while not SeqEngine.empty():

        # 得到下一个需要推理的序列
        tokens = SeqEngine.step()
        Logger.log(f"prompts tokens {tokens}")
        next_token = generator.only_prefill(tokens)
        Logger.log(f"new tokens{next_token}")
        # ipdb.set_trace()
        Logger.log(f"prompts：{tokens},next_token：{next_token}")
        if isinstance(next_token, torch.Tensor):
            next_token = next_token.tolist()
        new_tokens = []

        for i, token in enumerate(tokens):
            new_tokens.append(token.seq_tokens + next_token[i])  # 如果没有对应的新 token，保持原样
        Logger.log(f"new tokens：{new_tokens}")
        for token in new_tokens:
            res = generator.tokenizer.decode(token)
            Logger.log(f"生成的文本：{res}")
            SeqEngine.add_request(res)
        # pass
        # return
    
if __name__ == "__main__":
    fire.Fire(main)
