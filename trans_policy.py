# %%
# -*- coding: utf-8 -*-
import math

import os, csv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque

from episim import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    from torchvision import models as tvm
except Exception as e:
    tvm = None
    print("[WARN] torchvision을 불러오지 못했습니다. ResNet18 사용을 위해 torchvision 설치가 필요합니다.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TB_AVAILABLE = False
# -----------------------------
# Positional Encoding (sin/cos)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, E]
        return x + self.pe[: x.size(0)].unsqueeze(1)

# -----------------------------
# ResNet18 state encoder (C,H,W) -> d_model
# -----------------------------
class ResNet18StateEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        assert tvm is not None, "torchvision이 필요합니다. pip install torchvision"
        net = tvm.resnet18(weights=None)
        # conv1를 채널 수에 맞게 교체
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 마지막 fc를 d_model로 교체
        net.fc = nn.Linear(512, d_model)
        self.net = net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, C, H, W], uint8 or float
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs
        return self.net(x)  # [B, d_model]

# -----------------------------
# Pointer-style Transformer Policy (K fixed) + Value head
#  - 아이템 feature(h차원)를 받아 임베딩과 합성
# -----------------------------
class PointerTransformerPolicy(nn.Module):
    def __init__(
        self,
        n_items: int,
        obs_channels: int,
        item_feat_dim: int,   # h
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        use_item_layernorm: bool = True,
        t_scale : int = 180,
    ):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.obs_channels = obs_channels
        self.item_feat_dim = item_feat_dim
        self.t_scale = t_scale

        # Observation encoder: ResNet18 -> d_model
        self.state_encoder = ResNet18StateEncoder(obs_channels, d_model)

        # Item embeddings (learned index embedding)
        self.item_emb = nn.Embedding(n_items, d_model)

        # Item feature projection h -> d_model
        self.item_feat_proj = nn.Linear(item_feat_dim, d_model)
        self.item_ln = nn.LayerNorm(d_model) if use_item_layernorm else nn.Identity()

        # A learned state token; memory = [state_token+state_emb ; item_tokens]
        self.state_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer decoder (autoregressive)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        # self.pos_dec = PositionalEncoding(d_model, max_len=1024)

        # Critic (state value)
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model+1),  # +1 for time feature
            nn.Linear(d_model+1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Logit temperature
        self.log_temp = nn.Parameter(torch.zeros(1))

        # === PointerTransformerPolicy.__init__ 내부에 추가 ===
        self.stop_emb = nn.Parameter(torch.randn(1, self.d_model))  # [1, d]
        self.min_select_default = 0  # 필요하면 기본값

        # (선택) 정적 아이템 feature를 보관할 수 있음 (N, h)
        self.register_buffer("static_item_feats", None, persistent=False)

    # ---- optional: set static item features (N, h)
    def set_static_item_features(self, item_feats: torch.Tensor):
        """
        item_feats: [N, h] on same device as the model
        """
        assert item_feats.dim() == 2 and item_feats.size(0) == self.n_items and item_feats.size(1) == self.item_feat_dim
        self.static_item_feats = item_feats

    def _compose_item_tokens(self, B: int, device: torch.device, item_feats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        returns item_tokens: [N, B, d_model]
        item_feats can be:
          - None: use self.static_item_feats (N,h)
          - [N, h]: broadcast to B
          - [B, N, h]
        """
        base = self.item_emb.weight  # [N, d_model]

        if item_feats is None:
            assert self.static_item_feats is not None, "item_feats가 None이면 set_static_item_features로 설정해 주세요."
            feats = self.static_item_feats  # [N, h]
            feat_emb = self.item_feat_proj(feats)  # [N, d]
            tok = base + feat_emb  # [N, d]
            tok = self.item_ln(tok)
            tok = tok.unsqueeze(1).expand(-1, B, -1)  # [N,B,d]
            return tok

        if item_feats.dim() == 2:
            # [N,h]
            feat_emb = self.item_feat_proj(item_feats)  # [N,d]
            tok = base + feat_emb
            tok = self.item_ln(tok)
            tok = tok.unsqueeze(1).expand(-1, B, -1)  # [N,B,d]
            return tok

        elif item_feats.dim() == 3:
            # [B,N,h] -> [N,B,d]
            feat_emb = self.item_feat_proj(item_feats)  # [B,N,d]
            tok = base.unsqueeze(0) + feat_emb  # [B,N,d]
            tok = self.item_ln(tok)
            tok = tok.permute(1, 0, 2).contiguous()  # [N,B,d]
            return tok
        else:
            raise ValueError("item_feats shape must be [N,h] or [B,N,h]")
    def sincos_time(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        t: [B] (0~t_scale로 정규화하길 권장. 이미 0~1이면 t_scale=1)
        return: [B, d_model]  (pos-enc 과 동일한 주파수 구성)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)           # [B,1]
        half = d_model // 2
        # pos-enc와 동일한 주파수 스케일
        div = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32)
                        * (-math.log(10000.0) / half))
        x = (t / (self.t_scale + 1e-8)) * div  # [B, half]
        sin = torch.sin(x)
        cos = torch.cos(x)
        pe  = torch.cat([sin, cos], dim=-1)  # [B, 2*half]
        if pe.size(-1) < d_model:            # d_model이 홀수면 패딩
            pe = F.pad(pe, (0, d_model - pe.size(-1)))
        return pe  # [B,d_model]
    
    def build_memory(self, state_emb: torch.Tensor, times: torch.Tensor, item_feats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        state_emb: [B, d]
        returns memory: [1+N, B, d]
        """
        B = state_emb.size(0)
        device = state_emb.device
        state_tok = self.state_token.expand(-1, B, -1).clone()  # [1,B,d]
        state_tok[0] = state_tok[0] + state_emb                   # inject state
        time_tok = self.sincos_time(times, self.d_model).unsqueeze(0)  # [1,B,d]
        head = torch.cat([time_tok, state_tok], dim=0)            # [2,B,d]
        # time_vec = self.sincos_time(times, self.d_model, t_scale=t_scale)  # [B,d]
        # time_tok = time_vec.unsqueeze(0)                           # [1,B,d]
        # head = torch.cat([time_tok, state_tok], dim=0) 
        item_tok = self._compose_item_tokens(B, device, item_feats)  # [N,B,d]
        memory = torch.cat([head, item_tok], dim=0)        # [1+N,B,d]
        return memory

    def _step_decode(
        self,
        memory: torch.Tensor,                 # [1+N,B,d]
        prev_sel: List[torch.LongTensor],     # len=t, each [B]
        invalid_mask: Optional[torch.Tensor], # [B,N] bool
        greedy: bool = False,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        B = memory.size(1)
        device = memory.device

        if len(prev_sel) == 0:
            tgt = torch.zeros(1, B, self.d_model, device=device)  # START
        else:
            sel = torch.stack(prev_sel, dim=0)      # [t,B]
            tgt = self.item_emb(sel)                # [t,B,d]
        # tgt = self.pos_dec(tgt)

        T = tgt.size(0)
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)  # [T,B,d]
        query = dec_out[-1]  # [B,d]

        item_weights = self.item_emb.weight  # [N,d]
        temp = torch.exp(self.log_temp).clamp(0.05, 20.0)
        logits = (query @ item_weights.t()) / temp  # [B,N]

        # mask duplicates and externally invalid
        dup_mask = torch.zeros(B, self.n_items, dtype=torch.bool, device=device)
        if len(prev_sel) > 0:
            for s in prev_sel:
                dup_mask.scatter_(1, s.unsqueeze(1), True)
        if invalid_mask is not None:
            dup_mask = dup_mask | invalid_mask

        logits = logits.masked_fill(dup_mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if greedy:
            sel_idx = probs.argmax(dim=-1)
        else:
            sel_idx = dist.sample()
        logp = dist.log_prob(sel_idx)
        ent = dist.entropy()
        return sel_idx, logp, ent

    # @torch.no_grad()
    # def act(
    #     self,
    #     obs: torch.Tensor,                 # [B,C,H,W]
    #     K: int,
    #     item_feats: Optional[torch.Tensor] = None,  # None or [N,h] or [B,N,h]
    #     invalid_mask: Optional[torch.Tensor] = None,# [B,N] bool
    #     greedy: bool = False,
    # ):
    #     """
    #     Returns:
    #       action_multi_hot: [B,N] float (0/1)
    #       logprob_sum: [B]
    #       entropy_sum: [B]
    #       value: [B]
    #       selections: list length K of [B] Long
    #     """
    #     B = obs.size(0)
    #     state = self.state_encoder(obs)     # [B,d]
    #     memory = self.build_memory(state, item_feats)  # [1+N,B,d]
    #     V = self.value_head(state).squeeze(-1)         # [B]

    #     prev_sel: List[torch.LongTensor] = []
    #     logps: List[torch.Tensor] = []
    #     ents: List[torch.Tensor] = []
    #     sels: List[torch.Tensor] = []

    #     for _ in range(K):
    #         s, lp, en = self._step_decode(memory, prev_sel, invalid_mask, greedy=greedy)
    #         prev_sel.append(s)
    #         sels.append(s)
    #         logps.append(lp)
    #         ents.append(en)

    #     action = torch.zeros(B, self.n_items, dtype=torch.float32, device=obs.device)
    #     for s in sels:
    #         action.scatter_(1, s.unsqueeze(1), 1.0)

    #     logprob_mean = torch.stack(logps, dim=0).mean(dim=0)  # [B]
    #     entropy_mean = torch.stack(ents, dim=0).mean(dim=0)   # [B]
    #     return action, logprob_mean, entropy_mean, V, sels

    # def evaluate_actions(
    #     self,
    #     obs: torch.Tensor,                     # [B,C,H,W]
    #     selections: torch.Tensor,              # [B,K] Long  (teacher forcing)
    #     item_feats: Optional[torch.Tensor] = None,
    #     invalid_mask: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     PPO 업데이트용: 주어진 selections에 대한 logprob_sum, entropy_sum, value를 재계산
    #     """
    #     B, K = selections.size(0), selections.size(1)
    #     state = self.state_encoder(obs)                      # [B,d]
    #     memory = self.build_memory(state, item_feats)        # [1+N,B,d]
    #     V = self.value_head(state).squeeze(-1)               # [B]

    #     prev_sel: List[torch.LongTensor] = []
    #     logps, ents = [], []
    #     for t in range(K):
    #         # at step t, evaluate dist conditioned on prev selections, then take logp of selections[:,t]
    #         s_idx = selections[:, t]  # [B]
    #         # forward one step to get dist
    #         if len(prev_sel) == 0:
    #             tgt = torch.zeros(1, B, self.d_model, device=obs.device)
    #         else:
    #             sel = torch.stack(prev_sel, dim=0)
    #             tgt = self.item_emb(sel)
    #         tgt = self.pos_dec(tgt)

    #         T = tgt.size(0)
    #         causal = torch.triu(torch.ones(T, T, device=obs.device), diagonal=1).bool()
    #         dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
    #         query = dec_out[-1]  # [B,d]
    #         item_weights = self.item_emb.weight  # [N,d]
    #         temp = torch.exp(self.log_temp).clamp(0.05, 20.0)
    #         logits = (query @ item_weights.t()) / temp  # [B,N]

    #         dup_mask = torch.zeros(B, self.n_items, dtype=torch.bool, device=obs.device)
    #         if len(prev_sel) > 0:
    #             for s in prev_sel:
    #                 dup_mask.scatter_(1, s.unsqueeze(1), True)
    #         if invalid_mask is not None:
    #             dup_mask = dup_mask | invalid_mask
    #         logits = logits.masked_fill(dup_mask, -1e9)

    #         probs = F.softmax(logits, dim=-1)
    #         dist = torch.distributions.Categorical(probs=probs)
    #         logp = dist.log_prob(s_idx)
    #         ent = dist.entropy()

    #         logps.append(logp)
    #         ents.append(ent)
    #         prev_sel.append(s_idx)

    #     logprob_mean = torch.stack(logps, dim=0).mean(dim=0)
    #     entropy_mean = torch.stack(ents, dim=0).mean(dim=0)
    #     return logprob_mean, entropy_mean, V
    
    # === PointerTransformerPolicy에 유틸 메서드 추가 ===
    def _ext_weights(self) -> torch.Tensor:
        """
        [N+1, d] = [item_emb; stop_emb]
        """
        return torch.cat([self.item_emb.weight, self.stop_emb], dim=0)  # [N+1, d]

    def _build_step_mask(
        self,
        prev_sel: List[torch.LongTensor],   # 길이 t, 각 [B]
        invalid_item_mask: Optional[torch.Tensor],  # [B,N] or None
        t: int,
        min_select: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        반환: [B, N+1] bool mask (True=선택 불가). 마지막 인덱스가 STOP.
        - 이미 고른 아이템은 다시 못 고름
        - t < min_select 이면 STOP 금지
        - 어떤 batch 샘플이 이전 단계에서 STOP을 뽑았으면 (teacher-forcing 시),
        그 샘플은 이후 단계에선 아이템 금지, STOP만 허용(=분포가 STOP에 집중)
        """
        B = prev_sel[0].size(0) if len(prev_sel) > 0 else (invalid_item_mask.size(0) if invalid_item_mask is not None else 1)
        N = self.n_items
        STOP = N

        mask = torch.zeros(B, N + 1, dtype=torch.bool, device=device)

        # 이미 선택된 아이템 금지
        if len(prev_sel) > 0:
            sel_stack = torch.stack(prev_sel, dim=0)  # [t,B]
            # 아이템(0..N-1)만 금지 (STOP은 금지하지 않음)
            for s in prev_sel:
                # s in [0..N]일 수 있음(STOP=N)
                item_mask = s.clamp_max(N-1)  # STOP일 때는 N-1로 clamp되지만 아래에서 보정
                mask.scatter_(1, s.unsqueeze(1).clamp_max(N-1), True)  # 일단 찍고
            # STOP은 다시 금지하지 않음 (필요 시 금지 가능)
            # 아래에서 STOP에 대해 다시 열어둠
            mask[:, STOP] = False

            # 이미 STOP을 고른 샘플: 이후에는 아이템 금지, STOP만 허용
            stopped = (sel_stack == STOP).any(dim=0)  # [B] bool
            if stopped.any():
                mask[stopped, :N] = True    # 아이템 모두 금지
                mask[stopped, STOP] = False # STOP만 허용

        # 외부 invalid 아이템 마스크 반영
        if invalid_item_mask is not None:
            mask[:, :N] |= invalid_item_mask

        # min_select 도달 전에는 STOP 금지
        if t < min_select:
            mask[:, STOP] = True

        return mask  # [B, N+1]

    @torch.no_grad()
    def act_stop(
        self,
        obs: torch.Tensor,                    # [B,C,H,W]
        times: torch.Tensor,
        max_decisions: int,                   # K_max (상한)
        item_feats: Optional[torch.Tensor] = None,   # [N,h] 또는 [B,N,h] (옵션)
        invalid_item_mask: Optional[torch.Tensor] = None,  # [B,N] (옵션)
        min_select: Optional[int] = None,     # 최소 선택 개수, 기본 0
        greedy: bool = False,
    ):
        """
        STOP 토큰으로 가변 K를 선택.
        반환:
        action_multi_hot: [B,N] float (STOP 미포함)
        logprob_mean:     [B]    (실제 의사결정 길이에 대해 평균)
        entropy_mean:     [B]
        value:            [B]
        selections:       [B, max_decisions] (각 step의 선택 index, 0..N=STOP)
        lengths:          [B] (의사결정 길이; STOP을 포함한 길이. STOP 안 나오면 max_decisions)
        """
        
        B = obs.size(0)
        device = obs.device
        N = self.n_items
        STOP = N
        min_select = self.min_select_default if (min_select is None) else min_select

        #여기만 함
        state = self.state_encoder(obs)                       # [B,d]
        memory = self.build_memory(state, times, item_feats) if "item_feats" in self.build_memory.__code__.co_varnames \
                else self.build_memory(state, times)                # 기존/신규 시그니처 호환
        # print(state.shape)
        # print(times.shape)
        V = self.value_head(torch.cat((state, times), dim=-1)).squeeze(-1)                # [B]

        # step-by-step
        prev_sel: List[torch.LongTensor] = []
        step_logps, step_ents, step_sel = [], [], []
        first_stop_t = torch.full((B,), fill_value=-1, dtype=torch.long, device=device)

        W_ext = self._ext_weights()  # [N+1, d]

        for t in range(max_decisions):
            # 디코더 입력
            if t == 0:
                tgt = torch.zeros(1, B, self.d_model, device=device)
            else:
                sel = torch.stack(prev_sel, dim=0)  # [t,B]
                tgt = self.item_emb(sel.clamp_max(N-1))  # STOP은 임베딩이 없으므로 clamp로 아이템 임베딩만 사용
            # tgt = self.pos_dec(tgt)
            causal = torch.triu(torch.ones(tgt.size(0), tgt.size(0), device=device), diagonal=1).bool()
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            query = dec_out[-1]  # [B,d]

            # 점수: 아이템 N + STOP 1
            temp = torch.exp(self.log_temp).clamp(0.3, 20.0)
            logits_ext = (query @ W_ext.t()) / temp  # [B,N+1]

            # 마스크
            mask_ext = self._build_step_mask(prev_sel, invalid_item_mask, t, min_select, device)  # [B,N+1]
            logits_ext = logits_ext.masked_fill(mask_ext, -1e9)

            probs = F.softmax(logits_ext, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sel_idx = probs.argmax(dim=-1) if greedy else dist.sample()
            logp = dist.log_prob(sel_idx)
            ent = dist.entropy()

            # 기록
            prev_sel.append(sel_idx)
            step_sel.append(sel_idx)
            step_logps.append(logp)
            step_ents.append(ent)

            # STOP 처음 나온 위치 기록 (샘플별)
            just_stopped = (first_stop_t < 0) & (sel_idx == STOP)
            if just_stopped.any():
                first_stop_t[just_stopped] = t

            # 배치 내 일부가 STOP을 골라도, 나머지는 계속 진행해야 하므로
            # 루프는 끝까지(max_decisions) 돌고, 길이 마스크로 학습 시 평균을 낼 것임.

        # selections [B, max_decisions]
        selections = torch.stack(step_sel, dim=1)  # [B,Lmax]
        # 길이(의사결정 수): STOP 포함. STOP이 없으면 Lmax
        lengths = torch.where(first_stop_t >= 0, first_stop_t + 1, torch.full_like(first_stop_t, max_decisions))

        # 평균 logp/entropy (길이 마스크)
        # mask_t[b,t] = 1 if t < lengths[b]
        t_idx = torch.arange(max_decisions, device=device).unsqueeze(0)  # [1,Lmax]
        mask = (t_idx < lengths.unsqueeze(1)).float()  # [B,Lmax]

        logp_mat = torch.stack(step_logps, dim=1)  # [B,Lmax]
        ent_mat  = torch.stack(step_ents,  dim=1)  # [B,Lmax]

        denom = lengths.clamp(min=1).float()
        logprob_mean = (logp_mat * mask).sum(dim=1) / denom  # [B]
        entropy_mean = (ent_mat  * mask).sum(dim=1) / denom  # [B]

        # 액션 벡터(멀티바이너리): STOP 제외 & 길이 내에서만 1로 세팅
        action = torch.zeros(B, N, dtype=torch.float32, device=device)
        # selections[:,:length] 중 STOP이 아닌 인덱스만 1로
        sel_items = selections.clamp_max(N-1)                    # STOP=N → N-1로 임시 clamp
        is_item   = selections < N                               # [B,Lmax]
        # mask와 is_item를 같이 사용
        write_mask = (t_idx < lengths.unsqueeze(1)) & is_item    # [B,Lmax]
        if write_mask.any():
            # 배치 scatter: 각 t에 대해 scatter_를 수행
            for t in range(max_decisions):
                m = write_mask[:, t]  # [B]
                if m.any():
                    action[m] = action[m].scatter(1, sel_items[m, t].unsqueeze(1), 1.0)

        return action, logprob_mean, entropy_mean, V, selections, lengths

    def evaluate_actions_stop(
        self,
        obs: torch.Tensor,                     # [B,C,H,W]
        times: torch.Tensor,
        selections: torch.Tensor,              # [B,Lmax] (0..N=STOP)
        lengths: torch.Tensor,                 # [B] (STOP 포함 길이; STOP 없으면 Lmax)
        item_feats: Optional[torch.Tensor] = None,
        invalid_item_mask: Optional[torch.Tensor] = None,
        min_select: Optional[int] = None,
    ):
        """
        PPO 업데이트용: 주어진 (selections, lengths) 에 대해 teacher-forcing으로
        단계별 logp를 계산한 뒤, 길이 마스킹 평균을 반환.
        """
        device = obs.device
        B, Lmax = selections.size(0), selections.size(1)
        N = self.n_items
        STOP = N
        min_select = self.min_select_default if (min_select is None) else min_select

        state = self.state_encoder(obs)
        memory = self.build_memory(state, times, item_feats) if "item_feats" in self.build_memory.__code__.co_varnames \
                else self.build_memory(state, times)
        # print(state.shape, times.shape)
        V = self.value_head(torch.cat((state, times), dim=-1)).squeeze(-1)

        prev_sel: List[torch.LongTensor] = []
        step_logps, step_ents = [], []
        W_ext = self._ext_weights()  # [N+1,d]

        for t in range(Lmax):
            # 디코더 입력
            if t == 0:
                tgt = torch.zeros(1, B, self.d_model, device=device)
            else:
                sel = torch.stack(prev_sel, dim=0)
                tgt = self.item_emb(sel.clamp_max(N-1))
            # tgt = self.pos_dec(tgt)
            causal = torch.triu(torch.ones(tgt.size(0), tgt.size(0), device=device), diagonal=1).bool()
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            query = dec_out[-1]

            # 분포
            temp = torch.exp(self.log_temp).clamp(0.05, 20.0)
            logits_ext = (query @ W_ext.t()) / temp  # [B,N+1]

            mask_ext = self._build_step_mask(prev_sel, invalid_item_mask, t, min_select, device)
            logits_ext = logits_ext.masked_fill(mask_ext, -1e9)

            probs = F.softmax(logits_ext, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            s_t = selections[:, t]  # [B] (0..N)
            logp_t = dist.log_prob(s_t)
            ent_t  = dist.entropy()

            step_logps.append(logp_t)
            step_ents.append(ent_t)
            prev_sel.append(s_t)

        # 길이 마스킹 평균
        t_idx = torch.arange(Lmax, device=device).unsqueeze(0)  # [1,Lmax]
        mask = (t_idx < lengths.unsqueeze(1)).float()           # [B,Lmax]
        logp_mat = torch.stack(step_logps, dim=1)               # [B,Lmax]
        ent_mat  = torch.stack(step_ents,  dim=1)               # [B,Lmax]
        denom = lengths.clamp(min=1).float()

        logprob_mean = (logp_mat * mask).sum(dim=1) / denom
        entropy_mean = (ent_mat  * mask).sum(dim=1) / denom
        return logprob_mean, entropy_mean, V    
