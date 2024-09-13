from dataclasses import dataclass
from seq_manager import Request
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
import threading
import time

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
    seq_list:list[Request]  
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
            cls.seq_list.append(Request(seq, seq_id))
            # 序列到list的映射
            cls.seq_id_to_list[seq_id] = len(cls.seq_list) - 1


    # 静态方法：处理成token,以批次的形式返回tokens
    @classmethod
    def step(cls) -> list[Request]:
        if cls.seq_list:
            # 这里可以实现批次调度，随机一个请求
            import random
            
            temp = [cls.seq_list[0]]
            cls.seq_list = cls.seq_list[1:]  # 移除第一个请求
            return temp
        else:
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
    def update(cls,seq_id,length,layers_id,kv_pos):
        idx = cls.seq_id_to_list[seq_id]
        # cls.seq_list[idx]
        now_pos = cls.seq_list[idx].seq_begin_pos 
        for i in range(length):
            cls.seq_list[idx].seq_tokenpos_layer_id_to_kv_cache_id[(i+now_pos,layers_id)] = kv_pos
            pass
        
        cls.seq_list[idx].seq_begin_pos += length
    
    # 如果为空返回true
    @classmethod
    def empty(cls):
        if len(cls.seq_list) == 0:
            return True
        return False