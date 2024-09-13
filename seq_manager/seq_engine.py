from dataclasses import dataclass
from .request import Request
from llama.tokenizer import Tokenizer
import threading
import time
from typing import List, Dict
# 创建一个线程锁对象
lock = threading.Lock()

# 定义装饰器，用于同步函数
def synchronized(func):
    def wrapper(*args, **kwargs):
        with lock:
            result = func(*args, **kwargs)
            return result
    return wrapper
# 实现seq矿建
# 提供api add_request step
# 
class SeqEngine:
    # 初始化类变量
    seq_list:List[Request]  
    tokenizer = None
    cnt = 0
    seq_id_to_list : dict[int:int]

    @classmethod
    def initialize(cls, tokenizer: Tokenizer):
        """用于初始化类中的tokenizer"""
        cls.tokenizer = tokenizer
        cls.seq_list =[]
        cls.cnt = 0
        cls.seq_id_to_list = {}

    # 静态方法：添加request
    @classmethod
    def add_request(cls, prompts: list[str]):

        """通过静态方式添加request"""
        
        # 确保tokenizer已初始化
        if cls.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call initialize first.")
        
        # 添加prompt
        for prompt in prompts:
            # 编码prompt
            seq = cls.tokenizer.encode(prompt, True, False)
            # 获取唯一的seq_id
            seq_id = cls._get_seq_id()

            # 将请求添加到队列
            cls.seq_list.append(Request(seq, seq_id,prompt))
            
            # 序列到list的映射，后续只需要传seq_id即可
            cls.seq_id_to_list[seq_id] = len(cls.seq_list) - 1


    # 静态方法：处理成token,以批次的形式返回tokens
    @classmethod
    def step(cls) -> list[Request]:
        if cls.seq_list:
            # 这里可以实现批次调度，随机一个请求
            import random
            # todo 目前实现随机传输一个idx
            if True:
                # todo
                pass
                idx = random.randint(1,cls.cnt)
            temp = [idx]
            # cls.seq_list = cls.seq_list[1:]  # 移除第一个请求
            return temp
        else:
            # 不做处理
            return []

    # 静态方法：获取同步的seq_id
    @classmethod
    @synchronized
    def _get_seq_id(cls) -> int:
        """返回一个唯一的序列ID"""
        cls.cnt += 1
        return cls.cnt
    
    @classmethod
    @synchronized
    def update(cls,seq_id,i,layers_id,kv_pos):
        # SeqEngine.update(seq_id,i,layer,idx)
        idx = cls.seq_id_to_list[seq_id]
        # cls.seq_list[idx]
        # for i in range(length):
        cls.seq_list[idx].seq_tokenpos_layer_id_to_kv_cache_id[i][layers_id] = kv_pos
            # pass
        # cls.seq_list[idx].seq_begin_pos 
    
    @classmethod
    @synchronized
    def update_begin_pos(cls,seq_id):
        idx = cls.seq_id_to_list[seq_id]
        cls.seq_list[idx].seq_length += 1
        cls.seq_list[idx].seq_begin_pos = len(cls.seq_list[idx].seq_tokens)
        cls.seq_list[idx].seq_tokenpos_layer_id_to_kv_cache_id.extend([[0]*32])
    # 如果为空返回true
    @classmethod
    def empty(cls):
        if len(cls.seq_list) == 0:
            return True
        return False