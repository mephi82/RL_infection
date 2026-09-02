from trans_policy import *


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
# PPO Trainer with GAE
# -----------------------------

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
    t_scale: int = 180
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

        # self.last_obs = None
        # self.last_infos = None

    def close(self):
        self.logger.close()

    def set_static_item_features(self, feats_np: np.ndarray):
        feats = torch.from_numpy(feats_np).float().to(self.device)
        self.policy.set_static_item_features(feats)
        self.static_item_feats_t = feats

    def _torchify(self, obs_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs_np).to(self.device)  # uint8; policy 내부에서 /255

    def _gather_ep_stats(self, infos: dict):
        mask = infos.get("_episode", None)
        episode = infos.get("episode", None)
        if mask is not None and episode is not None and np.any(mask):
            r = episode["r"][mask]; l = episode["l"][mask]
            # self.ep_returns = r
            # self.ep_lengths = l
            for ri, li in zip(r, l):
                self.ep_returns.append(float(ri))
                self.ep_lengths.append(int(li))
                if len(self.ep_returns) > 100:  # 제한
                    self.ep_returns.pop(0); self.ep_lengths.pop(0)

    @torch.no_grad()
    def _policy_act_batch(self, obs_np: np.ndarray, infos):
        obs_t = self._torchify(obs_np)
        times_t = self._torchify(infos['time']).unsqueeze(1)
        action, logp_mean, ent_mean, V, sels, lens = self.policy.act_stop(
            obs_t,
            times_t,
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
    #### 이어 돌리기
    # def collect_rollout(self):
    #     cfg, B, Lmax = self.cfg, self.B, self.cfg.K_max

    #     # ---- 이어서 돌리기 모드: 첫 호출만 reset, 이후엔 계속 ----
    #     if self.last_obs is None:
    #         obs, infos = self.envs.reset()
    #     else:
    #         obs, infos = self.last_obs, (self.last_infos or {})

    #     T = cfg.num_steps
    #     obs_buf  = np.zeros((T,B,self.C,self.H,self.W), dtype=np.uint8)
    #     times_buf = np.zeros((T,B), dtype=np.int64)
    #     logp_buf = np.zeros((T,B), dtype=np.float32)
    #     val_buf  = np.zeros((T,B), dtype=np.float32)   # raw values
    #     rew_buf  = np.zeros((T,B), dtype=np.float32)
    #     done_buf = np.zeros((T,B), dtype=np.bool_)
    #     seq_buf  = np.zeros((T,B,Lmax), dtype=np.int64)  # selections with STOP 포함
    #     len_buf  = np.zeros((T,B), dtype=np.int64)

    #     # 시작 시점에도 에피소드가 막 끝났을 수 있으므로 통계 반영
    #     if infos:
    #         self._gather_ep_stats(infos)

    #     for t in range(T):
    #         obs_buf[t] = obs
    #         actions, logp_mean, values, seqs, lens = self._policy_act_batch(obs, infos)

    #         next_obs, rews, terms, truncs, infos = self.envs.step(actions)
    #         dones = np.logical_or(terms, truncs)
    #         times_buf[t] = infos['time']
    #         logp_buf[t] = logp_mean
    #         val_buf[t]  = values
    #         rew_buf[t]  = rews
    #         done_buf[t] = dones
    #         seq_buf[t]  = seqs
    #         len_buf[t]  = lens

    #         self._gather_ep_stats(infos)
    #         obs = next_obs

    #     # 다음 업데이트를 위해 마지막 상태 저장 (이어돌리기 핵심)
    #     self.last_obs = obs
    #     self.last_infos = infos

    #     # ---- 로깅용 raw 통계 ----
    #     raw_return_mean = float(np.sum(rew_buf, axis=0).mean())
    #     raw_return_std  = float(np.sum(rew_buf, axis=0).std())
    #     raw_reward_mean = float(rew_buf.mean())
    #     raw_reward_std  = float(rew_buf.std())

    #     # ---- 부트스트랩 값 ----
    #     obs_t = self._torchify(obs)
    #     time_t = self._torchify(infos['time']).unsqueeze(1)
    #     with torch.no_grad():
    #         V_last = self.policy.value_head(torch.cat((self.policy.state_encoder(obs_t), 
    #                                                    time_t), dim=-1)).squeeze(-1).cpu().numpy()  # [B]


    #     # ---- GAE (raw value 기준) ----
    #     adv = np.zeros_like(rew_buf, dtype=np.float32)
    #     lastgaelam = np.zeros(B, dtype=np.float32)
    #     for t in reversed(range(T)):
    #         if t == T - 1:
    #             next_nonterm = 1.0 - done_buf[t].astype(np.float32)
    #             next_value = V_last
    #         else:
    #             next_nonterm = 1.0 - done_buf[t + 1].astype(np.float32)
    #             next_value = val_buf[t + 1]
    #         delta = rew_buf[t] + cfg.gamma * next_nonterm * next_value - val_buf[t]
    #         lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * lastgaelam
    #         adv[t] = lastgaelam
    #     ret = adv + val_buf  # raw returns

    #     # ---- 정규화 ----
    #     adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    #     if cfg.use_return_rms_for_value:
    #         flat_rets = ret.reshape(-1)
    #         self.ret_rms.update(flat_rets)
    #         ret_mean, ret_std = float(self.ret_rms.mean), float(self.ret_rms.std)
    #         ret_norm = (ret - ret_mean) / ret_std
    #         val_norm = (val_buf - ret_mean) / ret_std
    #     else:
    #         ret_norm = ret
    #         val_norm = val_buf
    #         ret_mean, ret_std = 0.0, 1.0

    #     batch = dict( 
    #         obs=obs_buf, times=times_buf, act_seq=seq_buf, act_len=len_buf,
    #         old_logp=logp_buf, old_val=val_norm, adv=adv, ret=ret_norm,
    #         ret_mean=ret_mean, ret_std=ret_std,
    #     )
    #     rollout_info = {
    #         "ep/return_mean": (float(np.mean(self.ep_returns)) if len(self.ep_returns) else np.nan),
    #         "ep/return_std":  (float(np.std(self.ep_returns))  if len(self.ep_returns) else np.nan),
    #         "ep/len_mean":    (float(np.mean(self.ep_lengths)) if len(self.ep_lengths) else np.nan),
    #         "rollout/done_frac": float(done_buf.mean()),
    #         "raw/return_mean": raw_return_mean,
    #         "raw/return_std":  raw_return_std,
    #         "raw/reward_mean": raw_reward_mean,
    #         "raw/reward_std":  raw_reward_std,
    #         "raw/act_len_mean": float(len_buf.mean()),
    #         "raw/act_len_std": float(len_buf.std()),                
    #     }
    #     return batch, rollout_info
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


    #### 매번 리셋
    def collect_rollout(self):
        cfg, B, Lmax = self.cfg, self.B, self.cfg.K_max
        obs, infos = self.envs.reset()
        T = cfg.num_steps

        obs_buf  = np.zeros((T,B,self.C,self.H,self.W), dtype=np.uint8)
        times_buf = np.zeros((T,B), dtype=np.int64)
        logp_buf = np.zeros((T,B), dtype=np.float32)
        val_buf  = np.zeros((T,B), dtype=np.float32)   # raw values
        rew_buf  = np.zeros((T,B), dtype=np.float32)
        done_buf = np.zeros((T,B), dtype=np.bool_)
        seq_buf  = np.zeros((T,B,Lmax), dtype=np.int64)  # selections with STOP included
        len_buf  = np.zeros((T,B), dtype=np.int64)

        self._gather_ep_stats(infos)

        for t in range(T):
            
            obs_buf[t] = obs
            actions, logp_mean, values, seqs, lens = self._policy_act_batch(obs, infos)

            next_obs, rews, terms, truncs, infos = self.envs.step(actions)
            dones = np.logical_or(terms, truncs)

            times_buf[t] = infos['time']
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
        obs_t = self._torchify(obs)
        time_t = self._torchify(infos['time']).unsqueeze(1)
        with torch.no_grad():
            # value only: use state_encoder + value_head directly
            V_last = self.policy.value_head(torch.cat((self.policy.state_encoder(obs_t), 
                                                       time_t), dim=-1)).squeeze(-1).cpu().numpy()  # [B]
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
            delta = 0.01*rew_buf[t] + cfg.gamma * next_nonterm * next_value - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * lastgaelam
            adv[t] = lastgaelam
        # --------- GAE (raw values) ----------
        # ret = raw advantage + raw value
        ret = adv + val_buf

        # --------- normalize (advantages always; returns optionally for value) ----------
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if cfg.use_return_rms_for_value:
            flat_rets = ret.reshape(-1)
            self.ret_rms.update(flat_rets)                       # μ, σ는 항상 'raw return'에서 갱신
            ret_mean, ret_std = float(self.ret_rms.mean), float(self.ret_rms.std)
            ret_std = max(ret_std, 1e-8)                         # eps
            # ✅ 여기서 '한 번만' 정규화해서 배치에 담는다
            ret_norm = (ret - ret_mean) / ret_std
            val_norm = (val_buf - ret_mean) / ret_std
        else:
            ret_norm = ret
            val_norm = val_buf
            ret_mean, ret_std = 0.0, 1.0

        batch = dict(
            obs = obs_buf,
            times = times_buf,
            act_seq = seq_buf,
            act_len = len_buf,
            old_logp = logp_buf,
            old_val = val_norm,      # ✅ 정규화된 값(옵션)
            adv = adv,
            ret = ret_norm,          # ✅ 정규화된 값(옵션)
            ret_mean = ret_mean,     # update에서 v만 맞출 때 사용
            ret_std = ret_std,
        )

        rollout_info = {
            "ep/return_mean": (float(np.mean(self.ep_returns)) if len(self.ep_returns) else np.nan),
            "ep/return_std":  (float(np.std(self.ep_returns))  if len(self.ep_returns) else np.nan),
            "ep/len_mean":    (float(np.mean(self.ep_lengths)) if len(self.ep_lengths) else np.nan),
            "advantage/mean":   float(adv.mean()),
            "advantage/max":   float(adv.max()),
            "advantage/min":   float(adv.min()),
            # "rollout/done_frac": float(done_buf.mean()),
            "raw/return_mean": raw_return_mean,
            "raw/return_std":  raw_return_std,
            "raw/reward_mean": raw_reward_mean,
            "raw/reward_std":  raw_reward_std,
            "act/len_mean": float(len_buf.mean()),
            "act/len_std": float(len_buf.std()),
        }
        return batch, rollout_info


    def update(self, batch, update_idx: int):
        cfg = self.cfg
        # ----- CPU 텐서로 유지 (reshape만 TB로) -----
        obs_cpu  = torch.from_numpy(batch["obs"]).view(-1, self.C, self.H, self.W)            # [TB,C,H,W] (uint8, CPU)
        times_cpu= torch.from_numpy(batch["times"]).view(-1).long()                       # [TB,1]
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

        for nepoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, TB, cfg.minibatch_size):
                mb_idx_np = inds[start:start+cfg.minibatch_size]
                mb_idx = torch.as_tensor(mb_idx_np, dtype=torch.long)

                # ----- 이 미니배치에 필요한 것만 GPU로 -----
                obs_mb  = obs_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                times_mb  = times_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                seq_mb  = seq_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                len_mb  = len_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                olp_mb  = olp_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                oval_mb = oval_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                adv_mb  = adv_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)
                ret_mb  = ret_cpu.index_select(0, mb_idx).to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    new_logp, ent, v = self.policy.evaluate_actions_stop(
                        obs_mb, times_mb.unsqueeze(1), seq_mb, len_mb,
                        item_feats=self.static_item_feats_t, invalid_item_mask=None, min_select=cfg.min_select
                    )
                    # ✅ 배치의 old_val/ret가 정규화되어 있으므로 v도 '같은 μ,σ'로만 1회 정규화
                    if cfg.use_return_rms_for_value:
                        v = (v - ret_mean) / (ret_std + 1e-8)

                    ratio = torch.exp(new_logp - olp_mb)
                    pg1 = ratio * adv_mb
                    pg2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_mb
                    policy_loss = -torch.min(pg1, pg2).mean()

                    # value clipping은 '같은 스케일'에서 비교
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
            "train/entropy": total["ent"]/max(1,ns),
            "train/approx_kl": total["kl"]/max(1,ns),
            "train/clipfrac": total["clipfrac"]/max(1,ns),
            "train/grad_norm": total["grad"]/max(1,ns),
            "train/explained_variance_before": ev_before,
            "train/early_stop_kl": float(nepoch)+1,
        }
        return stats


    def train(self, total_updates: int, progress_fn=None):
        self.total_updates = total_updates
        steps_per_update = self.B * self.cfg.num_steps
        self.envs.reset()
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
                "train/lr": current_lr, **stats, **rinfo, 
            }
            self.logger.log(payload, step=self.global_steps)

            print(f"[upd {u:04d} {(roll_s+upd_s):.0f}] steps={self.global_steps} fps={fps:.0f} "
                  f"ret_mean={rinfo['ep/return_mean']:.3f} len_mean={rinfo['ep/len_mean']:.1f} "
                  f"pol={stats['loss/policy']:.3f} val={stats['loss/value']:.3f} "
                  f"ent={stats['train/entropy']:.3f} kl={stats['train/approx_kl']:.4f} clipfrac={stats['train/clipfrac']:.2f}"
                  )


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
def make_episim_env(kwargs):
    def thunk():
        env = EpiSimEnvironment(**kwargs)
        # env.reset(seed=seed + idx)
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
    N_ENVS = 24   # 8~32부터 시도 추천 (CPU/모델 속도에 따라 조절)
    K_MAX = 80
    C = len(STATES)

    env_params = {
        'city': city,
        'max_epis_length': 180,
        'ext_rate': 1/1000/7,
        'mean_recover': 5,
        'risk_coexist': 0.03,
        'n_visit_tries': 2,
        'r0': 100,
        'coef_block': 0.06,
        'coef_infect': 0.1,
        'gamma': 1.0
    }   
    # 4) 러너 생성 & 학습
    cfg = PPOVecConfig(
        K_max = K_MAX,                 # STOP 기반 최대 선택 길이 (상한)
        start_lr = 1e-5,
        end_lr = 1e-6,
        warmup_updates = 10,
        gamma = 1.0,
        clip_coef = 0.3,
        vf_coef = 0.05,
        ent_coef = 1e-5,
        max_grad_norm = 0.5,
        d_model = 256,
        t_scale = 180,
        update_epochs = 2,
        num_steps = 180,           # per update, per env
        minibatch_size = 128,
        logdir = "runs/episim_ppo_stop_timed",
        target_kl = 0.5,
        use_return_rms_for_value = True,   # True면 value를 normalized return에 맞춰 학습
        min_select = 0,                     # STOP 허용 전 최소 선택 수
        ckpt_dir = "checkpoints",
        save_every = 0,          # 0이면 주기 저장 비활성, >0이면 n 업데이트마다 저장
        best_metric_key = "ep/return_mean",  # 또는 "raw/return_mean"
        maximize_metric = True, # 값이 클수록 좋으면 True, 작을수록 좋으면 False
    )
    
    # 2) VectorEnv 만들기 (Async 권장)
    envs = gym.vector.AsyncVectorEnv(
        [make_episim_env(env_params) for i in range(N_ENVS)],
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

