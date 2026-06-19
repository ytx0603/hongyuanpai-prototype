# 红源派 · 技术原型

红源派三大核心技术模块的 PyTorch 实现，可在 RTX 4060 上完整运行。

## 模块概览

```
hongyuanpai-prototype/
├── 01_RLVR/              # RLVR 数字修复
│   ├── rlvr_train.py     # 主训练管线
│   ├── reward_fn.py      # 三维度奖励函数
│   ├── self_play.py      # 自我对弈循环
│   ├── grpo_optimizer.py # GRPO 策略优化
│   ├── sft_finetune.py   # SFT 微调
│   └── config.py
├── 02_3DCNN/             # 3D-CNN 智能监测
│   ├── pointcloud_pipeline.py
│   ├── model.py
│   ├── change_detection.py
│   ├── trend_prediction.py
│   └── config.py
├── 03_narrative_engine/  # 多模态叙事引擎
│   ├── kg_builder.py
│   ├── user_profiling.py
│   ├── rag_generator.py
│   ├── validator.py
│   ├── traceability.py
│   └── config.py
└── data/samples/         # 示例数据
```

## 环境要求

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scikit-learn transformers sentence-transformers faiss-cpu
```

## 运行

```bash
# RLVR 训练 demo（在 ModelNet40 模拟数据上）
python 01_RLVR/rlvr_train.py

# 3D-CNN 变化检测 demo
python 02_3DCNN/change_detection.py

# 叙事引擎 demo
python 03_narrative_engine/rag_generator.py
```
