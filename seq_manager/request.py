from dataclasses import dataclass
# 0 wait
# 1 prefill_running
# 2 prefill_unfinished
# 3 decode_running
# 4 prefill_finished
# 5 decode_finished
@dataclass
class Request:
    # 参数
    seq_id:int
    seq_length:int
    seq_state:int
    seq_finished_length:int
    seq_tokens:list[int]
    seq_begin_pos:int
    seq_generate_length:int
    seq_tokenpos_layer_id_to_kv_cache_id:dict
    seq_res:str
    
    # 构造
    def __init__(self,sq:list[int],seq_id:int,res:str):
        self.seq_begin_pos = 0
        self.seq_id = seq_id
        self.seq_length = len(sq)
        self.seq_state = 0
        self.seq_finished_length = 0
        self.seq_tokens = sq
        self.seq_generate_length = 1024  # 新添加的初始化
        # 序列内容
        self.seq_res = res
        # 映射层到不同层到kvcache的映射
        self.seq_tokenpos_layer_id_to_kv_cache_id = [[0]*32 for i in range(self.seq_length)]
    # def _get_id():
    #     return 1