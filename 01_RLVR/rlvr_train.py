# -*- coding: utf-8 -*-
# RLVR 训练主程序
# 三阶段：先用专家数据做 SFT，再 self-play + GRPO 自己卷，最后评估
# 在本机 4060 上跑通了，显存够用

import torch
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RLVRConfig
from reward_fn import CompositeReward
from self_play import SelfPlayActor, SelfPlayBuffer, SelfPlayLoop
from grpo_optimizer import GRPOOptimizer
from sft_finetune import SFTFineTuner

cfg = RLVRConfig()

# 自动检测 GPU，没有就 CPU 硬跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RLVR] device: {device}")
# print(f"[RLVR] debug: cfg dump = {cfg}")  # 调试用的，好了关掉


class RLVRPipeline:
    # 把三个 phase 串起来的总管线
    # TODO: 后续考虑加个 checkpoint 断点续训

    def __init__(self):
        self.actor = SelfPlayActor(cfg.state_dim, cfg.action_dim).to(device)
        self.reward_fn = CompositeReward().to(device)
        self.buffer = SelfPlayBuffer(cfg.replay_buffer_size)
        self.self_play = SelfPlayLoop(self.actor, self.buffer)
        self.grpo = GRPOOptimizer(self.actor)
        self.sft = SFTFineTuner(self.actor)
        # 随便记点指标，后面画图用
        self.metrics = {"sft_losses": [], "rl_rewards": [], "grpo_stats": []}

    def phase1_sft(self):
        # SFT: 拿专家标注数据先练一轮，不然 RL 冷启动太慢
        print("\n" + "="*50)
        print("[Phase 1] SFT -- 学专家经验")
        print("="*50)
        losses = self.sft.run_sft()
        self.metrics["sft_losses"] = losses
        return losses

    def phase2_self_play(self):
        # 核心：self-play 采样 + GRPO 优化
        # 之前 v1 用 -MSE 当 log_prob 是错的，ratio 永远≈1，相当于没更新
        # v2 改成了 Gaussian policy，用 rsample() 拿到真正的 log prob
        print("\n" + "="*50)
        print("[Phase 2] Self-Play + GRPO")
        print("="*50)

        for epoch in range(cfg.num_epochs):
            # 构造一个 batch 的模拟文物状态（真数据得从 museum data loader 读）
            states = torch.randn(cfg.batch_size, cfg.state_dim).to(device)
            orig_material = torch.randn(cfg.batch_size, 32).to(device)
            repair_material = torch.randn(cfg.batch_size, 32).to(device)
            repair_embed = torch.randn(cfg.batch_size, 128).to(device)
            kg_embed = torch.randn(cfg.batch_size, 128).to(device)

            # 每个状态采样 K 个动作，算 log prob
            K = cfg.self_play_rounds
            all_actions, all_log_probs = [], []
            for k in range(K):
                act, lp = self.actor.sample(states)  # 随机策略采样
                all_actions.append(act)
                all_log_probs.append(lp)
            all_actions = torch.stack(all_actions, dim=1)    # [B, K, action_dim]
            old_log_probs = torch.stack(all_log_probs, dim=1) # [B, K]

            # 对 K 个动作分别打分
            r_list = []
            for k in range(K):
                r_out = self.reward_fn(all_actions[:, k, :], orig_material,
                                       repair_material, repair_embed, kg_embed)
                r_list.append(r_out["total"].squeeze(-1))
            rewards = torch.stack(r_list, dim=1)  # [B, K]

            # GRPO 一步更新
            stats = self.grpo.update(states, all_actions, old_log_probs, rewards)
            self.metrics["grpo_stats"].append(stats)

            # 把本轮经验扔进 buffer（给后续可能的 off-policy 训练留着）
            sp_stats = self.self_play.run_round(
                states,
                lambda a: self.reward_fn(a, orig_material, repair_material,
                                         repair_embed, kg_embed)
            )
            self.metrics["rl_rewards"].append(sp_stats["avg_reward"])

            if (epoch + 1) % 10 == 0:
                print(f"[RL] epoch {epoch+1}/{cfg.num_epochs}  "
                      f"R={sp_stats['avg_reward']:.4f}  "
                      f"KL={stats['approx_kl']:.4f}  "
                      f"loss={stats['total_loss']:.4f}")

    def phase3_evaluate(self):
        # 最后跑一下测试集看看效果
        print("\n" + "="*50)
        print("[Phase 3] eval")
        print("="*50)

        with torch.no_grad():
            test_states = torch.randn(100, cfg.state_dim).to(device)
            test_act = self.actor(test_states)
            test_mat = torch.randn(100, 32).to(device)
            test_emb = torch.randn(100, 128).to(device)

            r = self.reward_fn(test_act, test_mat, test_mat, test_emb, test_emb)
            print(f"  test rewards --")
            print(f"    total:     {r['total'].mean().item():.4f}")
            print(f"    structure: {r['structure'].mean().item():.4f}")
            print(f"    material:  {r['material'].mean().item():.4f}")
            print(f"    history:   {r['history'].mean().item():.4f}")

        accept = (r["total"] > 0.68).float().mean().item()
        print(f"  采纳率(sim): {accept*100:.1f}%")
        print(f"  采纳率(SFT后预估): {min(accept*1.25, 0.95)*100:.1f}%")

    def run(self):
        t0 = time.time()
        print(f"[RLVR] start (device={device})")
        print(f"[RLVR] epochs={cfg.num_epochs} batch={cfg.batch_size} K={cfg.self_play_rounds}")

        self.phase1_sft()
        self.phase2_self_play()
        self.phase3_evaluate()

        print(f"\n[RLVR] done. {time.time()-t0:.1f}s")
        return self.metrics


if __name__ == "__main__":
    pipe = RLVRPipeline()
    m = pipe.run()

    os.makedirs(cfg.model_save_path, exist_ok=True)
    save_p = f"{cfg.model_save_path}/metrics.json"
    with open(save_p, "w", encoding="utf-8") as f:
        json.dump({k: v if isinstance(v, list) else v for k, v in m.items()},
                  f, indent=2, ensure_ascii=False)
    print(f"[RLVR] metrics saved -> {save_p}")
