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

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.array(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

class PPOLogger:
    def __init__(self, logdir: str = "runs/episim_ppo_stop"):
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir) if TB_AVAILABLE else None
        self.csv_path = os.path.join(logdir, "metrics.csv")
        self.csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = None
        self._header = False

    def log(self, metrics: Dict[str, Any], step: int):
        # TB
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating)):
                    self.writer.add_scalar(k, float(v), step)
        # CSV
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self.csv_file, fieldnames=["step"] + list(metrics.keys()))
            if not self._header:
                self._csv_writer.writeheader(); self._header = True
        row = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                row[k] = float(v)
            else:
                row[k] = v
        self._csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        if self.writer is not None:
            self.writer.flush(); self.writer.close()
        self.csv_file.close()
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
    ):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.obs_channels = obs_channels
        self.item_feat_dim = item_feat_dim

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
        self.pos_dec = PositionalEncoding(d_model, max_len=1024)

        # Critic (state value)
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
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

    def build_memory(self, state_emb: torch.Tensor, item_feats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        state_emb: [B, d]
        returns memory: [1+N, B, d]
        """
        B = state_emb.size(0)
        device = state_emb.device
        state_tok = self.state_token.expand(-1, B, -1).clone()  # [1,B,d]
        state_tok[0] = state_tok[0] + state_emb                  # inject state
        item_tok = self._compose_item_tokens(B, device, item_feats)  # [N,B,d]
        memory = torch.cat([state_tok, item_tok], dim=0)        # [1+N,B,d]
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
        tgt = self.pos_dec(tgt)

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

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,                 # [B,C,H,W]
        K: int,
        item_feats: Optional[torch.Tensor] = None,  # None or [N,h] or [B,N,h]
        invalid_mask: Optional[torch.Tensor] = None,# [B,N] bool
        greedy: bool = False,
    ):
        """
        Returns:
          action_multi_hot: [B,N] float (0/1)
          logprob_sum: [B]
          entropy_sum: [B]
          value: [B]
          selections: list length K of [B] Long
        """
        B = obs.size(0)
        state = self.state_encoder(obs)     # [B,d]
        memory = self.build_memory(state, item_feats)  # [1+N,B,d]
        V = self.value_head(state).squeeze(-1)         # [B]

        prev_sel: List[torch.LongTensor] = []
        logps: List[torch.Tensor] = []
        ents: List[torch.Tensor] = []
        sels: List[torch.Tensor] = []

        for _ in range(K):
            s, lp, en = self._step_decode(memory, prev_sel, invalid_mask, greedy=greedy)
            prev_sel.append(s)
            sels.append(s)
            logps.append(lp)
            ents.append(en)

        action = torch.zeros(B, self.n_items, dtype=torch.float32, device=obs.device)
        for s in sels:
            action.scatter_(1, s.unsqueeze(1), 1.0)

        logprob_mean = torch.stack(logps, dim=0).mean(dim=0)  # [B]
        entropy_mean = torch.stack(ents, dim=0).mean(dim=0)   # [B]
        return action, logprob_mean, entropy_mean, V, sels

    def evaluate_actions(
        self,
        obs: torch.Tensor,                     # [B,C,H,W]
        selections: torch.Tensor,              # [B,K] Long  (teacher forcing)
        item_feats: Optional[torch.Tensor] = None,
        invalid_mask: Optional[torch.Tensor] = None,
    ):
        """
        PPO 업데이트용: 주어진 selections에 대한 logprob_sum, entropy_sum, value를 재계산
        """
        B, K = selections.size(0), selections.size(1)
        state = self.state_encoder(obs)                      # [B,d]
        memory = self.build_memory(state, item_feats)        # [1+N,B,d]
        V = self.value_head(state).squeeze(-1)               # [B]

        prev_sel: List[torch.LongTensor] = []
        logps, ents = [], []
        for t in range(K):
            # at step t, evaluate dist conditioned on prev selections, then take logp of selections[:,t]
            s_idx = selections[:, t]  # [B]
            # forward one step to get dist
            if len(prev_sel) == 0:
                tgt = torch.zeros(1, B, self.d_model, device=obs.device)
            else:
                sel = torch.stack(prev_sel, dim=0)
                tgt = self.item_emb(sel)
            tgt = self.pos_dec(tgt)

            T = tgt.size(0)
            causal = torch.triu(torch.ones(T, T, device=obs.device), diagonal=1).bool()
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            query = dec_out[-1]  # [B,d]
            item_weights = self.item_emb.weight  # [N,d]
            temp = torch.exp(self.log_temp).clamp(0.05, 20.0)
            logits = (query @ item_weights.t()) / temp  # [B,N]

            dup_mask = torch.zeros(B, self.n_items, dtype=torch.bool, device=obs.device)
            if len(prev_sel) > 0:
                for s in prev_sel:
                    dup_mask.scatter_(1, s.unsqueeze(1), True)
            if invalid_mask is not None:
                dup_mask = dup_mask | invalid_mask
            logits = logits.masked_fill(dup_mask, -1e9)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            logp = dist.log_prob(s_idx)
            ent = dist.entropy()

            logps.append(logp)
            ents.append(ent)
            prev_sel.append(s_idx)

        logprob_mean = torch.stack(logps, dim=0).mean(dim=0)
        entropy_mean = torch.stack(ents, dim=0).mean(dim=0)
        return logprob_mean, entropy_mean, V
    
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

        state = self.state_encoder(obs)                       # [B,d]
        memory = self.build_memory(state, item_feats) if "item_feats" in self.build_memory.__code__.co_varnames \
                else self.build_memory(state)                # 기존/신규 시그니처 호환
        V = self.value_head(state).squeeze(-1)                # [B]

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
            tgt = self.pos_dec(tgt)
            causal = torch.triu(torch.ones(tgt.size(0), tgt.size(0), device=device), diagonal=1).bool()
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
            query = dec_out[-1]  # [B,d]

            # 점수: 아이템 N + STOP 1
            temp = torch.exp(self.log_temp).clamp(0.05, 20.0)
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
        memory = self.build_memory(state, item_feats) if "item_feats" in self.build_memory.__code__.co_varnames \
                else self.build_memory(state)
        V = self.value_head(state).squeeze(-1)

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
            tgt = self.pos_dec(tgt)
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
# -----------------------------
# PPO Trainer with GAE
# -----------------------------
@dataclass
class PPOConfig:
    K: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.3
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    num_steps: int = 2048     # rollout steps per update
    minibatch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PPORunner:
    def __init__(self, env, n_items: int, obs_channels: int, item_feat_dim: int, cfg: PPOConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = PointerTransformerPolicy(
            n_items=n_items,
            obs_channels=obs_channels,
            item_feat_dim=item_feat_dim,
            d_model=256,
            nhead=8,
            num_decoder_layers=2,
            dropout=0.1,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, weight_decay=1e-4)

        # If you have static item features from env: shape (N,h)
        self.static_item_feats = None  # set via set_static_item_features()

    def set_static_item_features(self, feats_np: np.ndarray):
        feats = torch.from_numpy(feats_np).float().to(self.device)  # [N,h]
        self.policy.set_static_item_features(feats)
        self.static_item_feats = feats

    def _to_tensor_obs(self, obs_np):
        obs_t = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # [1,C,H,W]
        return obs_t

    def collect_rollout(self):
        cfg = self.cfg
        obs, _ = self.env.reset()
        obs_buf = []
        act_sel_buf = []   # [T,K] Long selections
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        for _ in range(cfg.num_steps):
            obs_t = self._to_tensor_obs(obs)
            with torch.no_grad():
                action, logp, ent, V, sels = self.policy.act(
                    obs_t, K=cfg.K, item_feats=self.static_item_feats, invalid_mask=None, greedy=False
                )
            # selections -> [K] Long
            sels_1d = torch.stack(sels, dim=0).squeeze(1).cpu()  # [K]
            action_np = action.squeeze(0).cpu().numpy().astype(np.int32)

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = bool(terminated or truncated)

            obs_buf.append(obs.copy())
            act_sel_buf.append(sels_1d.numpy())
            logp_buf.append(logp.item())
            val_buf.append(V.item())
            rew_buf.append(float(reward))
            done_buf.append(done)

            obs = next_obs
            if done:
                obs, _ = self.env.reset()

        # last value for GAE bootstrap
        last_obs_t = self._to_tensor_obs(obs)
        with torch.no_grad():
            _, _, _, last_v, _ = self.policy.act(last_obs_t, K=cfg.K, item_feats=self.static_item_feats, greedy=True)

        # Convert to tensors
        obs_t = torch.from_numpy(np.stack(obs_buf, axis=0)).to(self.device)           # [T,C,H,W]
        act_sel_t = torch.from_numpy(np.stack(act_sel_buf, axis=0)).long().to(self.device)  # [T,K]
        logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=self.device)      # [T]
        val_t = torch.tensor(val_buf, dtype=torch.float32, device=self.device)        # [T]
        rew_t = torch.tensor(rew_buf, dtype=torch.float32, device=self.device)        # [T]
        done_t = torch.tensor(done_buf, dtype=torch.float32, device=self.device)      # [T]
        last_v = last_v.squeeze(0).detach()                                           # scalar

        # GAE
        adv = torch.zeros_like(rew_t)
        last_gae = 0.0
        for t in reversed(range(cfg.num_steps)):
            next_nonterminal = 1.0 - (done_t[t])
            next_value = last_v if t == cfg.num_steps - 1 else val_t[t + 1]
            delta = rew_t[t] + cfg.gamma * next_nonterminal * next_value - val_t[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + val_t

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        batch = {
            "obs": obs_t,            # [T,C,H,W]
            "act_sel": act_sel_t,    # [T,K]
            "logp": logp_t,          # [T]
            "val": val_t,            # [T]
            "adv": adv,              # [T]
            "ret": returns,          # [T]
        }
        return batch

    def update(self, batch):
        cfg = self.cfg
        T = batch["obs"].size(0)
        inds = np.arange(T)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, T, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = inds[start:end]
                obs_mb = batch["obs"][mb_idx]
                act_sel_mb = batch["act_sel"][mb_idx]
                old_logp_mb = batch["logp"][mb_idx]
                adv_mb = batch["adv"][mb_idx]
                ret_mb = batch["ret"][mb_idx]
                old_v_mb = batch["val"][mb_idx]

                new_logp, ent, v = self.policy.evaluate_actions(
                    obs_mb, act_sel_mb, item_feats=self.static_item_feats, invalid_mask=None
                )

                # policy loss (clipped)
                ratio = torch.exp(new_logp - old_logp_mb)
                pg1 = ratio * adv_mb
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_mb
                policy_loss = -torch.min(pg1, pg2).mean()

                # value loss (clip)
                v_clipped = old_v_mb + (v - old_v_mb).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss1 = (v - ret_mb).pow(2)
                v_loss2 = (v_clipped - ret_mb).pow(2)
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                entropy_loss = -ent.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

    def train(self, total_updates: int, progress_fn=None):
        for up in range(total_updates):
            batch = self.collect_rollout()
            self.update(batch)
            if progress_fn is not None:
                with torch.no_grad():
                    avg_ret = batch["ret"].mean().item()
                progress_fn(up, avg_ret)

# class AutoResetVectorEnv(gym.vector.VectorWrapper):
#     """
#     VectorEnv에서 step 후 done된 환경만 reset하고, 그 관측을 obs에 바로 채워 넣는 래퍼.
#     Gymnasium의 reset_done()이 반환하는 관측은 'done True인 env 인덱스 순서'와 일치한다고 가정.
#     """
#     def step(self, actions):
#         obs, rew, terminated, truncated, infos = self.env.step(actions)
#         dones = np.logical_or(terminated, truncated)
#         if np.any(dones):
#             reset_obs, reset_infos = self.env.reset_done()
#             obs[dones] = reset_obs  # done env들의 관측을 새로 채움
#             # 필요하면 infos["final_observation"], infos["final_info"] 저장 로직 추가 가능
#         return obs, rew, terminated, truncated, infos
    
# # -----------------------------
# # Config
# # -----------------------------
# @dataclass
# class PPOVecConfig:
#     K: int = 10
#     start_lr: float = 3e-4
#     end_lr: float = 1e-4
#     warmup_updates: int = 10 # warmup steps for lr
#     gamma: float = 0.99
#     gae_lambda: float = 0.95
#     clip_coef: float = 0.2
#     vf_coef: float = 0.5
#     ent_coef: float = 1e-3
#     max_grad_norm: float = 1.0
#     update_epochs: int = 4
#     num_steps: int = 1024        # rollout horizon per update (per env)
#     minibatch_size: int = 1024
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     logdir: str = "runs/episim_ppo"
#     target_kl: float = 0.02  # 예: 0.02 (선택)

# # -----------------------------
# # 벡터 PPO 러너 (Gymnasium 최신 API)
# # -----------------------------
# class PPORunnerVec:
#     """
#     - envs: VectorEnv (예: AsyncVectorEnv), 권장: gym.wrappers.vector.RecordEpisodeStatistics로 감싸기
#     - policy: PointerTransformerPolicy (ResNet-18 + Transformer Pointer, K 고정)
#     """
#     def __init__(
#         self,
#         envs: gym.vector.VectorEnv,
#         n_items: int,
#         obs_channels: int,
#         item_feat_dim: int,
#         cfg: PPOVecConfig,
#         policy_kwargs: Optional[dict] = None,
#     ):
#         self.envs = envs
#         self.cfg = cfg
#         self.B = envs.num_envs
#         self.device = torch.device(cfg.device)

#         kw = dict(
#             n_items=n_items,
#             obs_channels=obs_channels,
#             item_feat_dim=item_feat_dim,
#             d_model=256,
#             nhead=8,
#             num_decoder_layers=2,
#             dropout=0.1,
#         )
#         if policy_kwargs:
#             kw.update(policy_kwargs)

#         self.policy = PointerTransformerPolicy(**kw).to(self.device)
#         self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.start_lr, weight_decay=1e-4)

#                 # linear lr schedule with warmup
#         def lr_lambda(update):
#         # warmup then linear decay
#             if update < cfg.warmup_updates:
#                 return max(1e-8, update / float(cfg.warmup_updates))
#             progress = (update - cfg.warmup_updates) / max(1, (self.total_updates - cfg.warmup_updates))
#             lr = cfg.end_lr + (cfg.start_lr - cfg.end_lr) * (1.0 - progress)
#             return lr / cfg.start_lr
#         self.total_updates = 1_000 # 나중에 train()에서 설정
#         self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

#         self.logger = PPOLogger(cfg.logdir)
#         self.global_steps = 0
#         self.global_updates = 0

#         self.ep_returns: deque = deque(maxlen=100)
#         self.ep_lengths: deque = deque(maxlen=100)

#         obs_space = envs.single_observation_space
#         assert isinstance(obs_space, gym.spaces.Box)
#         self.C, self.H, self.W = obs_space.shape

#         # 정적 아이템 피처(선택): [N,h]
#         self.static_item_feats_t: Optional[torch.Tensor] = None
#         self.ret_rms = RunningMeanStd(shape=())

#     def close(self):
#         self.logger.close()

#     def set_static_item_features(self, feats_np: np.ndarray):
#         feats = torch.from_numpy(feats_np).float().to(self.device)  # [N,h]
#         self.policy.set_static_item_features(feats)
#         self.static_item_feats_t = feats

#     def _torchify_obs(self, obs_np: np.ndarray) -> torch.Tensor:
#         # [B,C,H,W] (uint8) -> torch on device (ResNet 내부에서 /255 처리)
#         return torch.from_numpy(obs_np).to(self.device)

#     @torch.no_grad()
#     def _policy_act_batch(self, obs_np: np.ndarray):
#         obs_t = self._torchify_obs(obs_np)
#         action, logp, ent, V, sels = self.policy.act(
#             obs_t, K=self.cfg.K, item_feats=self.static_item_feats_t, invalid_mask=None, greedy=False
#         )
        
#         sels_bt = torch.stack(sels, dim=1)  # [B,K]
#         return (
#             action.detach().cpu().numpy().astype(np.int8),  # [B,N] {0,1}
#             logp.detach().cpu().numpy(),                    # [B]
#             V.detach().cpu().numpy(),                       # [B]
#             sels_bt.detach().cpu().numpy(),                 # [B,K]
#         )

#     def _gather_episode_stats(self, infos):
#         """
#         RecordEpisodeStatistics(벡터) 기준:
#         - dict-of-arrays 형식이면: infos["_episode"] (bool mask), infos["episode"]["r"/"l"]
#         - DictInfoToList를 썼다면: list-of-dicts, 각 원소에 "episode"가 존재하는 인덱스만 집계
#         - 안전망: final_info 경로도 체크
#         """
#         # 1) dict-of-arrays
#         if isinstance(infos, dict):
#             mask = infos.get("_episode", None)
#             episode = infos.get("episode", None)
#             collected = 0
#             if mask is not None and episode is not None:
#                 m = np.asarray(mask, dtype=bool)
#                 if m.any():
#                     r_arr = np.asarray(episode.get("r", []), dtype=object)
#                     l_arr = np.asarray(episode.get("l", []), dtype=object)
#                     # r_arr, l_arr가 object일 수 있어 item()로 뽑음
#                     for i in np.where(m)[0]:
#                         ri = float(np.asarray(r_arr[i]).item()) if np.ndim(r_arr[i]) else float(r_arr[i])
#                         li = int(np.asarray(l_arr[i]).item()) if np.ndim(l_arr[i]) else int(l_arr[i])
#                         self.ep_returns.append(ri); self.ep_lengths.append(li)
#                         collected += 1
#             # 1-보: final_info도 확인
#             fi = infos.get("final_info", None)
#             if fi is not None:
#                 # final_info는 object 배열([None or dict])인 경우가 많음
#                 fi_arr = np.asarray(fi, dtype=object)
#                 for x in fi_arr:
#                     if isinstance(x, dict) and "episode" in x:
#                         e = x["episode"]
#                         if "r" in e and "l" in e:
#                             self.ep_returns.append(float(e["r"])); self.ep_lengths.append(int(e["l"]))
#                             collected += 1
#             return

#         # 2) list-of-dicts (DictInfoToList를 바깥에 씌운 경우)
#         if isinstance(infos, list):
#             for info in infos:
#                 if not isinstance(info, dict):
#                     continue
#                 e = info.get("episode")
#                 if e is not None and "r" in e and "l" in e:
#                     self.ep_returns.append(float(e["r"])); self.ep_lengths.append(int(e["l"]))
#                 # 안전망: final_info 안에 episode가 있을 수도 있음
#                 fi = info.get("final_info")
#                 if isinstance(fi, dict) and "episode" in fi:
#                     ee = fi["episode"]
#                     if "r" in ee and "l" in ee:
#                         self.ep_returns.append(float(ee["r"])); self.ep_lengths.append(int(ee["l"]))

#     def collect_rollout(self):
#         cfg, B = self.cfg, self.B
#         obs, infos = self.envs.reset()  # [B,C,H,W] uint8

#         T = cfg.num_steps
#         obs_buf   = np.zeros((T, B, self.C, self.H, self.W), dtype=np.uint8)
#         logp_buf  = np.zeros((T, B), dtype=np.float32)
#         val_buf   = np.zeros((T, B), dtype=np.float32)
#         rew_buf   = np.zeros((T, B), dtype=np.float32)
#         done_buf  = np.zeros((T, B), dtype=np.bool_)
#         sels_buf  = np.zeros((T, B, cfg.K), dtype=np.int64)

#         # 에피소드 통계 일부는 reset 이후 첫 step에서도 들어올 수 있으니, 초기 infos 처리
#         self._gather_episode_stats(infos)

#         for t in range(T):
#             obs_buf[t] = obs
#             actions, logp, values, sels = self._policy_act_batch(obs)
#             next_obs, rews, terms, truncs, infos = self.envs.step(actions)
#             dones = np.logical_or(terms, truncs)

#             logp_buf[t] = logp
#             val_buf[t]  = values
#             rew_buf[t]  = rews
#             done_buf[t] = dones
#             sels_buf[t] = sels

#             # 에피소드 통계 집계 (벡터 전용 키 사용)
#             self._gather_episode_stats(infos)

#             obs = next_obs  # NEXT_STEP autoreset 모드에서 그대로 사용

#         # === 여기서 raw reward 통계 먼저 계산 ===
#         raw_return_mean = float(np.sum(rew_buf, axis=0).mean())
#         raw_return_std  = float(np.sum(rew_buf, axis=0).std())
#         raw_reward_mean = float(rew_buf.mean())
#         raw_reward_std  = float(rew_buf.std())

#         # 부트스트랩 값 (마지막 관측)
#         with torch.no_grad():
#             _, _, last_V, _ = self._policy_act_batch(obs)  # [B]

#         # --------- GAE ----------
#         adv = np.zeros_like(rew_buf, dtype=np.float32)  # [T,B]
#         lastgaelam = np.zeros(B, dtype=np.float32)

#         for t in reversed(range(T)):
#             if t == T - 1:
#                 next_nonterminal = 1.0 - done_buf[t].astype(np.float32)
#                 next_value = last_V
#             else:
#                 next_nonterminal = 1.0 - done_buf[t + 1].astype(np.float32)
#                 next_value = val_buf[t + 1]

#             delta = rew_buf[t] + cfg.gamma * next_nonterminal * next_value - val_buf[t]
#             lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
#             adv[t] = lastgaelam

#         ret = adv + val_buf  # [T,B]

#         # === reward/return normalization ===
#         flat_rets = ret.reshape(-1)
#         self.ret_rms.update(flat_rets)
#         ret = (ret - self.ret_rms.mean) / self.ret_rms.std
#         val_buf = (val_buf - self.ret_rms.mean) / self.ret_rms.std

#         # 표준화
#         adv_mean = adv.mean()
#         adv_std = adv.std() + 1e-8
#         adv = (adv - adv_mean) / adv_std

#         batch = dict(
#             obs = obs_buf,      # [T,B,C,H,W] uint8
#             act_sel = sels_buf, # [T,B,K]
#             old_logp = logp_buf,# [T,B]
#             old_val = val_buf,  # [T,B]
#             adv = adv,          # [T,B]
#             ret = ret,          # [T,B]
#         )

#         rollout_info = {
#             "ep/return_mean": (np.mean(self.ep_returns) if len(self.ep_returns) else np.nan),
#             "ep/return_std": (np.std(self.ep_returns) if len(self.ep_returns) else np.nan),
#             "ep/len_mean": (np.mean(self.ep_lengths) if len(self.ep_lengths) else np.nan),
#             # raw reward/return 로깅
#             "raw/return_mean": raw_return_mean,
#             "raw/return_std":  raw_return_std,
#             "raw/reward_mean": raw_reward_mean,
#             "raw/reward_std":  raw_reward_std,
#         }
#         return batch, rollout_info

#     def _to_device_batch(self, batch):
#         # (T,B,...) -> (TB, ...)
#         def flat(x):
#             x = torch.from_numpy(x)
#             return x.view(-1, *x.shape[2:])
#         obs_tb  = flat(batch["obs"]).to(self.device)             # [TB,C,H,W]
#         act_tb  = flat(batch["act_sel"]).long().to(self.device)  # [TB,K]
#         logp_tb = flat(batch["old_logp"]).squeeze(-1).float().to(self.device)  # [TB]
#         val_tb  = flat(batch["old_val"]).squeeze(-1).float().to(self.device)   # [TB]
#         adv_tb  = flat(batch["adv"]).squeeze(-1).float().to(self.device)       # [TB]
#         ret_tb  = flat(batch["ret"]).squeeze(-1).float().to(self.device)       # [TB]
#         return obs_tb, act_tb, logp_tb, val_tb, adv_tb, ret_tb

#     @staticmethod
#     def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
#         var_y = torch.var(y_true)
#         if var_y.item() < 1e-8:
#             return np.nan
#         return (1 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

#     def update(self, batch):
#         cfg = self.cfg
#         obs_tb, act_tb, old_logp_tb, old_v_tb, adv_tb, ret_tb = self._to_device_batch(batch)

#         TB = obs_tb.size(0)
#         inds = np.arange(TB)

#         ev_before = self._explained_variance(old_v_tb, ret_tb)

#         total_loss = total_pol = total_val = total_ent = total_kl = total_clipfrac = total_gradnorm = 0.0
#         nsamples = 0
#         early_stop = False

#         for _ in range(cfg.update_epochs):
#             np.random.shuffle(inds)
#             for start in range(0, TB, cfg.minibatch_size):
#                 end = start + cfg.minibatch_size
#                 mb_idx = inds[start:end]
#                 n_mb = len(mb_idx)

#                 obs_mb = obs_tb[mb_idx]
#                 act_mb = act_tb[mb_idx]
#                 old_logp_mb = old_logp_tb[mb_idx]
#                 adv_mb = adv_tb[mb_idx]
#                 ret_mb = ret_tb[mb_idx]
#                 old_v_mb = old_v_tb[mb_idx]

#                 new_logp, ent, v = self.policy.evaluate_actions(
#                     obs_mb, act_mb, item_feats=self.static_item_feats_t, invalid_mask=None
#                 )

#                 ratio = torch.exp(new_logp - old_logp_mb)
#                 pg1 = ratio * adv_mb
#                 pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_mb
#                 policy_loss = -torch.min(pg1, pg2).mean()

#                 v_clipped = old_v_mb + (v - old_v_mb).clamp(-cfg.clip_coef, cfg.clip_coef)
#                 v_loss1 = (v - ret_mb).pow(2)
#                 v_loss2 = (v_clipped - ret_mb).pow(2)
#                 value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

#                 entropy_loss = -ent.mean()

#                 loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
#                 self.optimizer.step()

#                 with torch.no_grad():
#                     approx_kl = (old_logp_mb - new_logp).mean().abs()
#                     clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
#                     total_loss += loss.item() * n_mb
#                     total_pol  += policy_loss.item() * n_mb
#                     total_val  += value_loss.item() * n_mb
#                     total_ent  += ent.mean().item() * n_mb
#                     total_kl   += approx_kl.item() * n_mb
#                     total_clipfrac += clipfrac.item() * n_mb
#                     total_gradnorm += float(grad_norm) * n_mb
#                     nsamples += n_mb

#                 if cfg.target_kl is not None and approx_kl.item() > cfg.target_kl:
#                     early_stop = True
#                     break
#             if early_stop:
#                 break

#         stats = {
#             "loss/total": total_loss / max(1, nsamples),
#             "loss/policy": total_pol / max(1, nsamples),
#             "loss/value": total_val / max(1, nsamples),
#             "entropy": total_ent / max(1, nsamples),
#             "approx_kl": total_kl / max(1, nsamples),
#             "clipfrac": total_clipfrac / max(1, nsamples),
#             "grad_norm": total_gradnorm / max(1, nsamples),
#             "explained_variance_before": ev_before,
#             "early_stop_kl": float(early_stop),
#         }
#         # lr schedule step
#         self.lr_scheduler.step()
#         return stats

#     def save(self, path: str):
#         ckpt = {
#             "policy_state": self.policy.state_dict(),
#             "optimizer_state": self.optimizer.state_dict(),
#             "global_steps": self.global_steps,
#             "cfg": self.cfg.__dict__,  # 학습 config 저장해두면 재현성 좋음
#             "ret_rms": getattr(self, "ret_rms", None),  # reward normalizer 있으면 같이
#         }
#         torch.save(ckpt, path)
#         print(f"[PPO] 모델 저장 완료: {path}")

#     def load(self, path: str, map_location=None):
#         ckpt = torch.load(path, map_location=map_location)
#         self.policy.load_state_dict(ckpt["policy_state"])
#         self.optimizer.load_state_dict(ckpt["optimizer_state"])
#         self.global_steps = ckpt.get("global_steps", 0)
#         if "ret_rms" in ckpt and ckpt["ret_rms"] is not None:
#             self.ret_rms = ckpt["ret_rms"]
#         print(f"[PPO] 모델 로드 완료: {path} (steps={self.global_steps})")


#     def train(self, total_updates: int, progress_fn=None):
#         cfg = self.cfg
#         steps_per_update = self.B * cfg.num_steps
#         self.total_updates = total_updates
#         for u in range(total_updates):
#             t0 = time()
#             batch, rollout_info = self.collect_rollout()
#             rollout_sec = time() - t0

#             t1 = time()
#             update_stats = self.update(batch)
#             update_sec = time() - t1

#             self.global_steps += steps_per_update
#             self.global_updates += 1

#             fps = steps_per_update / max(1e-6, rollout_sec)
#             lr = self.lr_scheduler.get_last_lr()[0]

#             log_payload = {
#                 "env/num_envs": self.B,
#                 "env/horizon": cfg.num_steps,
#                 "time/rollout_s": rollout_sec,
#                 "time/update_s": update_sec,
#                 "time/fps": fps,
#                 "lr": lr,
#                 **update_stats,
#                 **rollout_info,
#                 "progress/steps": self.global_steps,
#             }
#             self.logger.log_scalars(log_payload, step=self.global_steps)

#             print(
#                 f"[upd {u:04d}] steps={self.global_steps} fps={fps:.0f} "
#                 f"ret_mean={rollout_info['ep/return_mean']:.3f} len_mean={rollout_info['ep/len_mean']:.1f} "
#                 f"pol={update_stats['loss/policy']:.3f} val={update_stats['loss/value']:.3f} "
#                 f"ent={update_stats['entropy']:.3f} kl={update_stats['approx_kl']:.4f} "
#                 f"clipfrac={update_stats['clipfrac']:.2f}"
#             )

#             if progress_fn is not None:
#                 progress_fn(u, rollout_info["ep/return_mean"])


@dataclass
class PPOVecConfig:
    K_max: int = 10                 # STOP 기반 최대 선택 길이 (상한)
    start_lr: float = 3e-4
    end_lr: float = 1e-4
    warmup_updates: int = 10
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.3
    vf_coef: float = 0.5
    ent_coef: float = 1e-2
    max_grad_norm: float = 1.0
    d_model: int = 256
    update_epochs: int = 4
    num_steps: int = 1024           # per update, per env
    minibatch_size: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logdir: str = "runs/episim_ppo_stop"
    target_kl: float = 0.2
    use_return_rms_for_value: bool = True   # True면 value를 normalized return에 맞춰 학습
    min_select: int = 0                     # STOP 허용 전 최소 선택 수
    ckpt_dir: str = "checkpoints"
    save_every: int = 0          # 0이면 주기 저장 비활성, >0이면 n 업데이트마다 저장
    best_metric_key: str = "ep/return_mean"  # 또는 "raw/return_mean"
    maximize_metric: bool = True # 값이 클수록 좋으면 True, 작을수록 좋으면 False

class PPORunnerVecStop:
    def __init__(self,
                 envs: gym.vector.VectorEnv,
                 n_items: int,
                 obs_channels: int,
                 item_feat_dim: int,
                 cfg: PPOVecConfig,
                 policy_kwargs: Optional[dict] = None):
        self.envs = envs
        self.cfg = cfg
        self.B = envs.num_envs
        self.device = torch.device(cfg.device)

        kw = dict(n_items=n_items, obs_channels=obs_channels, item_feat_dim=item_feat_dim,
                  d_model=cfg.d_model, nhead=8, num_decoder_layers=2, dropout=0.1)
        if policy_kwargs: kw.update(policy_kwargs)
        self.policy = PointerTransformerPolicy(**kw).to(self.device)

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.start_lr, weight_decay=1e-4)
        # scheduler with warmup + linear decay over total_updates (set later)
        self.total_updates = 1000
        def lr_lambda(u):
            if u < cfg.warmup_updates:
                return max(1e-8, u / float(max(1, cfg.warmup_updates)))
            prog = (u - cfg.warmup_updates) / max(1, (self.total_updates - cfg.warmup_updates))
            lr = cfg.end_lr + (cfg.start_lr - cfg.end_lr) * (1.0 - prog)
            return lr / cfg.start_lr
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # logger
        self.logger = PPOLogger(cfg.logdir)
        self.global_steps = 0

        # obs space
        obs_space = envs.single_observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        self.C, self.H, self.W = obs_space.shape

        # static item features [N,h]
        self.static_item_feats_t: Optional[torch.Tensor] = None

        # return running stats (for value scaling)
        self.ret_rms = RunningMeanStd(shape=())

        # roll episode stats
        self.ep_returns, self.ep_lengths = [], []

        self.best_score = -math.inf if self.cfg.maximize_metric else math.inf
        self.best_ckpt_path = os.path.join(self.cfg.ckpt_dir, "best.pt")

    def close(self):
        self.logger.close()

    def set_static_item_features(self, feats_np: np.ndarray):
        feats = torch.from_numpy(feats_np).float().to(self.device)
        self.policy.set_static_item_features(feats)
        self.static_item_feats_t = feats

    def _torch_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs_np).to(self.device)  # uint8; policy 내부에서 /255

    def _gather_ep_stats(self, infos: dict):
        mask = infos.get("_episode", None)
        episode = infos.get("episode", None)
        if mask is not None and episode is not None and np.any(mask):
            r = episode["r"][mask]; l = episode["l"][mask]
            for ri, li in zip(r, l):
                self.ep_returns.append(float(ri))
                self.ep_lengths.append(int(li))
                if len(self.ep_returns) > 1000:  # 제한
                    self.ep_returns.pop(0); self.ep_lengths.pop(0)

    @torch.no_grad()
    def _policy_act_batch(self, obs_np: np.ndarray):
        obs_t = self._torch_obs(obs_np)
        action, logp_mean, ent_mean, V, sels, lens = self.policy.act_stop(
            obs_t,
            max_decisions=self.cfg.K_max,
            item_feats=self.static_item_feats_t,
            invalid_item_mask=None,
            min_select=self.cfg.min_select,
            greedy=False,
        )
        return (action.cpu().numpy().astype(np.int8),
                logp_mean.cpu().numpy(),
                V.cpu().numpy(),
                sels.cpu().numpy().astype(np.int64),
                lens.cpu().numpy().astype(np.int64))

    def collect_rollout(self):
        cfg, B, Lmax = self.cfg, self.B, self.cfg.K_max
        obs, infos = self.envs.reset()
        T = cfg.num_steps

        obs_buf  = np.zeros((T,B,self.C,self.H,self.W), dtype=np.uint8)
        logp_buf = np.zeros((T,B), dtype=np.float32)
        val_buf  = np.zeros((T,B), dtype=np.float32)   # raw values
        rew_buf  = np.zeros((T,B), dtype=np.float32)
        done_buf = np.zeros((T,B), dtype=np.bool_)
        seq_buf  = np.zeros((T,B,Lmax), dtype=np.int64)  # selections with STOP included
        len_buf  = np.zeros((T,B), dtype=np.int64)

        self._gather_ep_stats(infos)

        for t in range(T):
            obs_buf[t] = obs
            actions, logp_mean, values, seqs, lens = self._policy_act_batch(obs)
            next_obs, rews, terms, truncs, infos = self.envs.step(actions)
            dones = np.logical_or(terms, truncs)

            logp_buf[t] = logp_mean
            val_buf[t]  = values
            rew_buf[t]  = rews
            done_buf[t] = dones
            seq_buf[t]  = seqs
            len_buf[t]  = lens

            self._gather_ep_stats(infos)
            obs = next_obs

        # raw reward/return stats for logging
        raw_return_mean = float(np.sum(rew_buf, axis=0).mean())
        raw_return_std  = float(np.sum(rew_buf, axis=0).std())
        raw_reward_mean = float(rew_buf.mean())
        raw_reward_std  = float(rew_buf.std())

        # bootstrap value (raw)
        obs_t = self._torch_obs(obs)
        with torch.no_grad():
            # value only: use state_encoder + value_head directly
            V_last = self.policy.value_head(self.policy.state_encoder(obs_t)).squeeze(-1).cpu().numpy()  # [B]

        # --------- GAE (raw values) ----------
        adv = np.zeros_like(rew_buf, dtype=np.float32)
        lastgaelam = np.zeros(B, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T-1:
                next_nonterm = 1.0 - done_buf[t].astype(np.float32)
                next_value = V_last
            else:
                next_nonterm = 1.0 - done_buf[t+1].astype(np.float32)
                next_value = val_buf[t+1]
            delta = rew_buf[t] + cfg.gamma * next_nonterm * next_value - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_buf  # raw returns

        # --------- normalize (advantages always; returns optionally for value) ----------
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if cfg.use_return_rms_for_value:
            flat_rets = ret.reshape(-1)
            self.ret_rms.update(flat_rets)
            ret_mean, ret_std = float(self.ret_rms.mean), float(self.ret_rms.std)
            ret_norm = (ret - ret_mean) / ret_std
            val_norm = (val_buf - ret_mean) / ret_std
            last_v_norm = (V_last - ret_mean) / ret_std
        else:
            ret_norm = ret
            val_norm = val_buf
            last_v_norm = V_last
            ret_mean, ret_std = 0.0, 1.0  # for logging consistency

        # pack
        batch = dict(
            obs = obs_buf,           # [T,B,C,H,W]
            act_seq = seq_buf,       # [T,B,Lmax]
            act_len = len_buf,       # [T,B]
            old_logp = logp_buf,     # [T,B] (mean logp)
            old_val = val_norm,      # [T,B] (normalized if enabled)
            adv = adv,               # [T,B] (std-normalized)
            ret = ret_norm,          # [T,B]
            ret_mean = ret_mean,     # scalars
            ret_std = ret_std,
        )

        rollout_info = {
            "ep/return_mean": (float(np.mean(self.ep_returns)) if len(self.ep_returns) else np.nan),
            "ep/return_std":  (float(np.std(self.ep_returns))  if len(self.ep_returns) else np.nan),
            "ep/len_mean":    (float(np.mean(self.ep_lengths)) if len(self.ep_lengths) else np.nan),
            "rollout/done_frac": float(done_buf.mean()),
            "raw/return_mean": raw_return_mean,
            "raw/return_std":  raw_return_std,
            "raw/reward_mean": raw_reward_mean,
            "raw/reward_std":  raw_reward_std,
            "raw/act_len_mean": float(len_buf.mean()),
            "raw/act_len_std": float(len_buf.std()),
        }
        return batch, rollout_info

    def _to_device_batch(self, batch):
        def flat(x):
            x = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
            return x.view(-1, *x.shape[2:])  # (T,B,...) -> (TB,...)
        obs_tb  = flat(batch["obs"]).to(self.device)            # [TB,C,H,W]
        seq_tb  = flat(batch["act_seq"]).long().to(self.device) # [TB,Lmax]
        len_tb  = flat(batch["act_len"]).long().to(self.device) # [TB]
        logp_tb = flat(batch["old_logp"]).squeeze(-1).float().to(self.device)
        val_tb  = flat(batch["old_val"]).squeeze(-1).float().to(self.device)
        adv_tb  = flat(batch["adv"]).squeeze(-1).float().to(self.device)
        ret_tb  = flat(batch["ret"]).squeeze(-1).float().to(self.device)
        ret_mean = float(batch["ret_mean"]); ret_std = float(batch["ret_std"])
        return obs_tb, seq_tb, len_tb, logp_tb, val_tb, adv_tb, ret_tb, ret_mean, ret_std

    @staticmethod
    def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        var_y = torch.var(y_true)
        if var_y.item() < 1e-8: return float("nan")
        return (1 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

    def update(self, batch, update_idx: int):
        cfg = self.cfg
        # ----- CPU 텐서로 유지 (reshape만 TB로) -----
        obs_cpu  = torch.from_numpy(batch["obs"]).view(-1, self.C, self.H, self.W)            # [TB,C,H,W] (uint8, CPU)
        seq_cpu  = torch.from_numpy(batch["act_seq"]).view(-1, self.cfg.K_max).long()         # [TB,Lmax]
        len_cpu  = torch.from_numpy(batch["act_len"]).view(-1).long()                         # [TB]
        olp_cpu  = torch.from_numpy(batch["old_logp"]).view(-1).float()                       # [TB]
        oval_cpu = torch.from_numpy(batch["old_val"]).view(-1).float()                        # [TB]
        adv_cpu  = torch.from_numpy(batch["adv"]).view(-1).float()                            # [TB]
        ret_cpu  = torch.from_numpy(batch["ret"]).view(-1).float()                            # [TB]
        ret_mean, ret_std = float(batch["ret_mean"]), float(batch["ret_std"])

        TB = obs_cpu.size(0)
        inds = np.arange(TB)
        ev_before = self._explained_variance(oval_cpu, ret_cpu)

        total = {"loss":0.0, "pol":0.0, "val":0.0, "ent":0.0, "kl":0.0, "clipfrac":0.0, "grad":0.0}
        ns, early_stop = 0, False

        # AMP 설정 (원하면 cfg.use_amp, cfg.amp_dtype 옵션화 가능)
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype==torch.float16))

        for _ in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, TB, cfg.minibatch_size):
                mb_idx_np = inds[start:start+cfg.minibatch_size]
                mb_idx = torch.as_tensor(mb_idx_np, dtype=torch.long)

                # ----- 이 미니배치에 필요한 것만 GPU로 -----
                obs_mb  = obs_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                seq_mb  = seq_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                len_mb  = len_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                olp_mb  = olp_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                oval_mb = oval_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                adv_mb  = adv_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                ret_mb  = ret_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    new_logp, ent, v = self.policy.evaluate_actions_stop(
                        obs_mb, seq_mb, len_mb,
                        item_feats=self.static_item_feats_t, invalid_item_mask=None, min_select=cfg.min_select
                    )
                    if cfg.use_return_rms_for_value:
                        v = (v - ret_mean) / (ret_std + 1e-8)

                    ratio = torch.exp(new_logp - olp_mb)
                    pg1 = ratio * adv_mb
                    pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_mb
                    policy_loss = -torch.min(pg1, pg2).mean()

                    v_clipped = oval_mb + (v - oval_mb).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss1 = (v - ret_mb).pow(2)
                    v_loss2 = (v_clipped - ret_mb).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    entropy_loss = -ent.mean()
                    loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    grad = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (olp_mb - new_logp).mean().abs()
                    clipfrac  = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
                    n = len(mb_idx_np)
                    total["loss"] += loss.item()*n; total["pol"] += policy_loss.item()*n
                    total["val"]  += value_loss.item()*n; total["ent"] += ent.mean().item()*n
                    total["kl"]   += approx_kl.item()*n; total["clipfrac"] += clipfrac.item()*n
                    total["grad"] += float(grad)*n; ns += n

                    if approx_kl.item() > cfg.target_kl:
                        early_stop = True
                        break
            if early_stop:
                break

        self.lr_scheduler.step()
        stats = {
            "loss/total": total["loss"]/max(1,ns),
            "loss/policy": total["pol"]/max(1,ns),
            "loss/value": total["val"]/max(1,ns),
            "entropy": total["ent"]/max(1,ns),
            "approx_kl": total["kl"]/max(1,ns),
            "clipfrac": total["clipfrac"]/max(1,ns),
            "grad_norm": total["grad"]/max(1,ns),
            "explained_variance_before": ev_before,
            "early_stop_kl": float(early_stop),
        }
        return stats


    def train(self, total_updates: int, progress_fn=None):
        self.total_updates = total_updates
        steps_per_update = self.B * self.cfg.num_steps

        for u in range(total_updates):
            t0 = time()
            batch, rinfo = self.collect_rollout()
            roll_s = time() - t0

            t1 = time()
            stats = self.update(batch, u)
            upd_s = time() - t1

            self.global_steps += steps_per_update
            fps = steps_per_update / max(1e-6, roll_s)
            current_lr = self.lr_scheduler.get_last_lr()[0]

            payload = {
                "env/num_envs": self.B, "env/horizon": self.cfg.num_steps,
                "time/rollout_s": roll_s, "time/update_s": upd_s, "time/fps": fps,
                "lr": current_lr, **stats, **rinfo, "progress/steps": self.global_steps,
            }
            self.logger.log(payload, step=self.global_steps)

            print(f"[upd {u:04d} {(roll_s+upd_s):.0f}] steps={self.global_steps} fps={fps:.0f} "
                  f"ret_mean={rinfo['ep/return_mean']:.3f} len_mean={rinfo['ep/len_mean']:.1f} "
                  f"pol={stats['loss/policy']:.3f} val={stats['loss/value']:.3f} "
                  f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} clipfrac={stats['clipfrac']:.2f}")


            # --- Save best / periodic checkpoints ---
            metric = payload.get(self.cfg.best_metric_key, float("nan"))
            is_valid = (metric == metric)  # not NaN
            is_better = (metric > self.best_score) if self.cfg.maximize_metric else (metric < self.best_score)

            if is_valid and is_better:
                self.best_score = metric
                self.save(self.best_ckpt_path)
                print(f"[CKPT] new best {self.cfg.best_metric_key}={metric:.4f} -> {self.best_ckpt_path}")

            if self.cfg.save_every and (u + 1) % self.cfg.save_every == 0:
                path = os.path.join(self.cfg.ckpt_dir, f"upd_{u+1:05d}.pt")
                self.save(path)

            if progress_fn:
                progress_fn(u, rinfo["ep/return_mean"])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_steps": self.global_steps,
            "cfg": self.cfg.__dict__,
            "ret_rms": {"mean": self.ret_rms.mean, "var": self.ret_rms.var, "count": self.ret_rms.count},
        }
        torch.save(ckpt, path)
        print(f"[CKPT] saved: {path}")

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_steps = ckpt.get("global_steps", 0)
        if "ret_rms" in ckpt:
            rms = ckpt["ret_rms"]
            self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count = rms["mean"], rms["var"], rms["count"]
        print(f"[CKPT] loaded: {path} (steps={self.global_steps})")            


# 1) EpiSim 환경 팩토리
def make_episim_env(seed: int, idx: int, kwargs):
    def thunk():
        env = EpiSimEnvironment(**kwargs)
        env.reset(seed=seed + idx)
        
        return env
    return thunk                 
# %%
# -----------------------------
# Dummy usage (치환해서 EpiSim에 연결)
# -----------------------------
if __name__ == "__main__":
    with open('city_local.pkl', 'rb') as f:
            city = pickle.load(f)

    item_feats = city.facs[['xcoor','ycoor','affiliated', 'visit', 'locality','risk']].values
    N, H_ITEM = item_feats.shape

    SEED = 42
    N_ENVS = 16   # 8~32부터 시도 추천 (CPU/모델 속도에 따라 조절)
    K_MAX = 80
    C = len(STATES)

    env_params = {
        'city': city,
        'max_epis_length': 180,
        'ext_rate': 1/1000/7,
        'mean_recover': 5,
        'risk_coexist': 0.03,
        'n_visit_tries': 2,
        'r0': 10,
        'coef_block': 0.06,
        'coef_infect': 0.1,
        'gamma': 1.0
    }   
    # 4) 러너 생성 & 학습
    cfg = PPOVecConfig(K_max=K_MAX, start_lr=1e-4, end_lr=5e-5, warmup_updates=10,
        clip_coef=0.3, ent_coef=1e-2, target_kl=0.5, num_steps=196, gamma=1.0,
        d_model=256,
        minibatch_size=128, update_epochs=4, device="cuda" if torch.cuda.is_available() else "cpu",
        use_return_rms_for_value=True, min_select=0,
    )
    
    # 2) VectorEnv 만들기 (Async 권장)
    envs = gym.vector.AsyncVectorEnv(
        [make_episim_env(SEED, i, env_params) for i in range(N_ENVS)],
        shared_memory=True,
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs, buffer_length=1000)
    print(f"ENVs MADE")
    # 3) 정적 아이템 feature 준비 (모든 env 공통이면 [N,h])
    item_feats = np.random.randn(N, H_ITEM).astype(np.float32)


    runner = PPORunnerVecStop(envs, n_items=N, obs_channels=C, item_feat_dim=H_ITEM, cfg=cfg)
    runner.set_static_item_features(item_feats)
    
    runner.train(total_updates=2000)

    # 학습이 끝나면
    runner.save("ppo_episim_ckpt.pt")

    # 나중에 재시작할 때
    # runner = PPORunnerVecStop(envs, n_items=N, obs_channels=C, item_feat_dim=H_ITEM, cfg=cfg)
    # runner.set_static_item_features(item_feats)
    # runner.load("ppo_episim_ckpt.pt")

    # 나중에 평가만 하고 싶을 때
    # runner.load("checkpoints/episim_stop/best.pt", map_location="cpu")
    # runner.policy.eval()
    # env에서 greedy 실행 등 수행

    # 5) 추론 예시
    obs, _ = envs.reset(seed=SEED)
    obs_t = torch.from_numpy(obs).to(cfg.device)  # [B,C,H,W]
    with torch.no_grad():
        act, logp, ent, V, sels = runner.policy.act(
            obs_t, K=cfg.K, item_feats=runner.static_item_feats_t, greedy=True
        )
    print("selection shape:", torch.stack(sels, dim=1).shape)  # [B,K]
    print("action shape   :", act.shape)  # [B,N]
    runner.close()
    envs.close()
# %%

