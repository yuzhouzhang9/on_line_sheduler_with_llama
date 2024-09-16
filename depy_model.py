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
from cache_manager import KVCacheManager
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
    # Logger.log(f"SeqEngine{SeqEngine}")
    # Logger.log(f"tokens:{SeqEngine.seq_list}")
    # ipdb.set_trace() 
    
    
    def check_prefill(tokens_idx:List[int]) -> bool:
        # Logger.log(f"SeqEngine.seq_id_to_list:{SeqEngine.seq_id_to_list}")
        pos = SeqEngine.seq_id_to_list[tokens_idx[0]]
        if SeqEngine.seq_list[pos].seq_state == 0:
            return True
        return False
    start_generation_time = time.time()
    while not SeqEngine.empty():

        # 得到下一个需要推理的序列，只需要传递seq_id即可
        tokens = SeqEngine.step()
        
        # 
        iteration_stage = 0
        
        # Logger.log(f"prompts tokens {tokens}")
        if check_prefill(tokens):
            # Logger.log(f"start prefiil stage")
            next_token = generator.only_prefill(tokens)
        else:
            iteration_stage = 1
            # Logger.log(f"start decode stage")
            next_token = generator.only_decode(tokens)
        # Logger.log(f"new tokens{next_token}")
        # ipdb.set_trace()
        # Logger.log(f"prompts：{tokens},next_token：{next_token}")
        if isinstance(next_token, torch.Tensor):
            next_token = next_token.tolist()
        
        for i, seq_id in enumerate(tokens):
            pos = SeqEngine.seq_id_to_list[seq_id]
            # Logger.log(f"SeqEngine:{SeqEngine}")
            # Logger.log(f"old Request{SeqEngine.seq_list[pos]}")
            #添加新的token    
            SeqEngine.seq_list[pos].seq_tokens.extend(next_token[i])  # 追加新 tokenx
            # 更新seq_begin_pos
            SeqEngine.update_begin_pos(seq_id)
            # udpate stage
            if iteration_stage == 0:
                SeqEngine.complete_prefill_stage(seq_id)
            elif iteration_stage  == 1:
                # SeqEngine.
                pass
            # elif 
            # 
            # Logger.log(f"new Request token length{SeqEngine.seq_list[pos].seq_length} , id {SeqEngine.seq_list[pos].seq_id} , has generate tokens{SeqEngine.seq_list[pos].seq_has_generate_tokens}")
            # res = generator.tokenizer.decode(SeqEngine.seq_list[pos].seq_tokens)
            # Logger.log(f"生成的文本：{res}")
            # new_tokens.append(token.seq_tokens + next_token[i])  # 如果没有对应的新 token，保持原样
    end_geration_time = time.time()
    Logger.log(f"total cost {end_geration_time - start_generation_time}")
    for res in SeqEngine.seq_list:
        res1 = generator.tokenizer.decode(res.seq_tokens)
        print(res1)
        Logger.log(f"idx:{res.seq_id} generate:{res1}")

if __name__ == "__main__":
    fire.Fire(main)
