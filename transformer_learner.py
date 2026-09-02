# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    from torchvision import models as tvm
except Exception as e:
    tvm = None
    print("[WARN] torchvision을 불러오지 못했습니다. ResNet18 사용을 위해 torchvision 설치가 필요합니다.")

from time import time
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
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        # Logit temperature
        self.log_temp = nn.Parameter(torch.zeros(1))

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

        logprob_sum = torch.stack(logps, dim=0).sum(dim=0)  # [B]
        entropy_sum = torch.stack(ents, dim=0).sum(dim=0)   # [B]
        return action, logprob_sum, entropy_sum, V, sels

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

        logprob_sum = torch.stack(logps, dim=0).sum(dim=0)
        entropy_sum = torch.stack(ents, dim=0).sum(dim=0)
        return logprob_sum, entropy_sum, V

# -----------------------------
# PPO Trainer with GAE
# -----------------------------
@dataclass
class PPOConfig:
    K: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
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
        timepoint = time()
        for up in range(total_updates):
            batch = self.collect_rollout()
            self.update(batch)
            if progress_fn is not None:
                with torch.no_grad():
                    avg_ret = batch["ret"].mean().item()
                progress_fn(up, avg_ret, time() - timepoint)

# -----------------------------
# Dummy usage (치환해서 EpiSim에 연결)
# -----------------------------
if __name__ == "__main__":
    import pickle
    from episim import *
    # 예시 환경 (EpiSimEnvironment로 교체)
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
    env = EpiSimEnvironment(city, max_epis_length=180, ext_rate=1/1000/7, 
                            mean_recover=5,
                            risk_coexist = 0.03, 
                            n_visit_tries = 2, r0=10,
                            coef_block=0.05, coef_infect=0.1, gamma=0.999)
    
    N = env.action_space.n  # MultiBinary(n)
    C, H, W = env.observation_space.shape

    # 아이템 feature 예시: (N, h)
    
    item_feats = city.facs[['xcoor','ycoor','affiliated', 'visit', 'locality','risk']].values
    _, h = item_feats.shape

    cfg = PPOConfig(K=10, num_steps=1024, minibatch_size=256, update_epochs=4, lr=3e-4)
    runner = PPORunner(env, n_items=N, obs_channels=C, item_feat_dim=h, cfg=cfg)
    runner.set_static_item_features(item_feats)

    def progress(u, avg_ret, elapse):
        print(f"[update {u:04d}] avg_return(batch)={avg_ret:.4f} elased={elapse:.1f}")

    # 짧게 돌려보기 (실전에서는 수백~수천 업데이트)
    runner.train(total_updates=10000, progress_fn=progress)

    # 추론 예시
    obs, _ = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(runner.cfg.device)
    with torch.no_grad():
        act, logp, ent, V, sels = runner.policy.act(
            obs_t, K=cfg.K, item_feats=runner.static_item_feats, greedy=True
        )
    print("selected indices:", [s.item() for s in sels])
    print("action sum:", int(act.sum().item()))



