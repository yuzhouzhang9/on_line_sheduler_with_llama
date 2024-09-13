import torch
from seq_manager import *
from logger import *

class KVCacheManager:
    cache_k = None
    cache_v = None
    cnt = 0

    # 
    @classmethod
    def initialize_cache(cls, max_batch_size, max_seq_len, layers, n_local_kv_heads, head_dim):
        max_batch_size = 1
        cls.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, layers, n_local_kv_heads, head_dim)
        ).cuda()
        cls.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, layers, n_local_kv_heads, head_dim)
        ).cuda()

    # 获取cache
    @classmethod
    def get_cache(cls, bsz, seq: list[Request], layer: int):
        batch_size = 1
        seq_length = seq[0].seq_begin_pos
        head_number = 32
        head_dim = 128
        
        key = torch.zeros(
            (
                batch_size, seq_length, head_number, head_dim
            )
        ).cuda()
        
        value = torch.zeros(
            (
                batch_size, seq_length, head_number, head_dim
            )
        ).cuda()
        for i in range(seq[0].seq_length):
            idx = seq[0].seq_tokenpos_layer_id_to_kv_cache_id[(i,layer)]
            key[0,i,layer] = cls.cache_k[0,idx,layer]
            value[0,i,layer] = cls.cache_v[0,idx,layer]
        return (key, value)

    # 
    @classmethod
    def reserved_cache(cls, key, value, seqs: list[Request], layer: int):
        # Logger.log(f"reserved cache for seqs{seqs},keys:{key},value:{value}")
        for idx,seq in enumerate(seqs):
            for i in range(seq.seq_length):
                idx = cls._get_idx()
                cls.cache_k[0,idx,layer] = key[0,i,layer]
                cls.cache_v[0,idx,layer] = value[0,i,layer]
                SeqEngine.update(seq.seq_id,i,layer,idx)
        return 

    # 
    @classmethod
    def _get_idx(cls):
        cls.cnt += 1
        return cls.cnt