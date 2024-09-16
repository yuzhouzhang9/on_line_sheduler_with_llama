# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import json
import os
import sys
import time
import ipdb
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from .mode_config import ModelArgs
from .model import Transformer
from .tokenizer import Tokenizer
from seq_manager.request import Request
from logger import *
from seq_manager.seq_engine import SeqEngine
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self. tokenizer = tokenizer
    
    # 对于所有批次生成一个token
    # 假定长度相同
    @torch.inference_mode()
    def only_prefill(
        self,
        # 传入的是seq_id
        requests:list[int],
        temperature:float=0.2,
        top_p:float=0.8,
    ) -> list[int]:
        
        # 批次和长度，
        if True:
            # todo
            mx_length = 0
            for seq_id in requests:
                pos = SeqEngine.seq_id_to_list[seq_id]
                mx_length = max(mx_length, len(SeqEngine.seq_list[pos].seq_tokens))
        batch_size = len(requests)

        # 预分配一个这么大的空间
        tokens = torch.zeros((batch_size, mx_length), dtype=torch.long, device="cuda")

        # 赋值
        for k, t in enumerate(requests):
            pos =  SeqEngine.seq_id_to_list[seq_id]
            tokens[k] = torch.tensor(SeqEngine.seq_list[pos].seq_tokens, dtype=torch.long, device="cuda")
        
        # 
        # Logger.log(f"prefill stage tokens:{tokens},prefill tensor shape:{tokens.shape}")
        
        # ipdb.set_trace()
        begin_prefill_time = time.time()
        # 一次forword以生产下一个token
        logits = self.model.forward(tokens,requests)
        end_prefill_time = time.time()
        Logger.log(f"prefille cost time {end_prefill_time - begin_prefill_time}")
        # loss = 
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        Logger.log(f"choose token cost {time.time() - end_prefill_time},generate one token cost {time.time() - begin_prefill_time}")
        return next_token
    
    @torch.inference_mode()
    def only_decode(
        self,
        requests:list[Request],
        temperature:float=0.2,
        top_p:float=0.8,
    ) -> list[int]:
        # 批次和长度，
        if True:
            # decode阶段
            # 默认tokens是1
            mx_length = 0
            for pos in requests:
                pos = SeqEngine.seq_id_to_list[pos]
                # Logger.log(f"decode stage pos:{pos},seq_begin_pos:{SeqEngine.seq_list[pos].seq_begin_pos},seq_tokens_len:{len(SeqEngine.seq_list[pos].seq_tokens)}")
                mx_length = max(len(SeqEngine.seq_list[pos].seq_tokens) - SeqEngine.seq_list[pos].seq_begin_pos,mx_length)
            batch_size = len(requests)
        # Logger.log(f"decode stage max_length:{mx_length},batch_size:{batch_size}")
        
        # 预分配一个这么大的空间
        tokens = torch.zeros((batch_size, mx_length), dtype=torch.long, device="cuda")
        
        # 赋值
        for k, t in enumerate(requests):
            pos = SeqEngine.seq_id_to_list[t]
            # Logger.log(f"tokens.shape{tokens.shape},{torch.tensor(SeqEngine.seq_list[pos].seq_tokens[t.seq_begin_pos:], dtype=torch.long, device="cuda").shape}")
            tokens[k] = torch.tensor(SeqEngine.seq_list[pos].seq_tokens[SeqEngine.seq_list[pos].seq_begin_pos:], dtype=torch.long, device="cuda")
        
        # 
        # Logger.log(f"decode stage tokens:{tokens},decode tensor shape:{tokens.shape}")

        # ipdb.set_trace()
        begin_decode_time = time.time()

        # 一次forword以生产下一个token
        logits = self.model.forward(tokens,requests)

        end_decode_time = time.time()
        Logger.log(f"iterator one token decode cost time {end_decode_time- begin_decode_time}")
        # loss = 
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        Logger.log(f"choose token cost {time.time() - end_decode_time},generate one token cost {time.time() - begin_decode_time}")
        return next_token
    
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
