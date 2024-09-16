import torch
from logger import Logger
from seq_manager import SeqEngine
from seq_manager.request import Request
# 
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
    def get_cache(cls, bsz, seqs: list[int], layer: int):
        batch_size = bsz
        mx_seq_length = 0
        for seq_id in seqs:
            pos = SeqEngine.seq_id_to_list[seq_id]
            mx_seq_length  = max(mx_seq_length,SeqEngine.seq_list[pos].seq_begin_pos)
        head_number = 32
        head_dim = 128
        key = torch.zeros(
            (
                batch_size, mx_seq_length,head_number, head_dim
            )
        ).cuda()
        value = torch.zeros(
            (
                batch_size, mx_seq_length, head_number, head_dim
            )
        ).cuda()
        for bz,seq_id in enumerate(seqs):
            pos = SeqEngine.seq_id_to_list[seq_id]
            # Logger.log(f"bz:b{bz},seq_id:{seq_id}")
            for i in range(SeqEngine.seq_list[pos].seq_begin_pos):
                # for j in range(32):
                j = layer
                idx = SeqEngine.seq_list[pos].seq_tokenpos_layer_id_to_kv_cache_id[i][j]
                key[bz,i] = cls.cache_k[0,idx,j]
                value[bz,i] = cls.cache_v[0,idx,j]
            # Logger.log(f"key:{key},value:{value}")
        return (key, value)

    @classmethod
    def check(cls,seqs:list[int]) -> bool:
        for seq_id in seqs:
            pos = SeqEngine.seq_id_to_list[seq_id]
            if SeqEngine.seq_list[pos].seq_begin_pos > 0:
                return True
        return False
    
    # 
    @classmethod
    def reserved_cache(cls, key, value, seqs: list[int], layer: int):
        # Logger.log(f"reserved cache for seqs{seqs},keys:{key},value:{value}")
        for idx,seq_id in enumerate(seqs):
            pos = SeqEngine.seq_id_to_list[seq_id]
            for i in range(SeqEngine.seq_list[pos].seq_begin_pos,len(SeqEngine.seq_list[pos].seq_tokens)):
                idx = cls._get_idx()
                cls.cache_k[0,idx,layer] = key[0,i - SeqEngine.seq_list[pos].seq_begin_pos ,layer]
                cls.cache_v[0,idx,layer] = value[0,i - SeqEngine.seq_list[pos].seq_begin_pos,layer]
                SeqEngine.update(seq_id,i,layer,idx)
            
        return   
    @classmethod
    def update_seq_begin_pos(cls,seqs:list[int]):
        for seq_id in seqs:
            SeqEngine.update_begin_pos(seq_id)
    # 
    @classmethod
    def _get_idx(cls):
        cls.cnt += 1
        return cls.cnt