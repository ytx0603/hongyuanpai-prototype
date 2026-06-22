# 红源π · AI赋能红色文物保护

> **三大核心技术**：RLVR 数字修复 · 3D-CNN 智能监测 · 多模态叙事引擎

## 📁 项目结构

```
hongyuanpai-prototype/
├── common/                    # 公共模块（数据接口抽象层）
│   ├── __init__.py
│   └── data_interface.py      # DataProvider 抽象接口 + SyntheticDataProvider
├── 01_RLVR/                   # 模块一：RLVR 数字修复（v2 随机策略）
│   ├── rlvr_train.py          #   主训练管线（SFT → Self-Play → GRPO）
│   ├── reward_fn.py           #   三维度奖励函数（结构/材料/历史）
│   ├── self_play.py           #   自我对弈 + 高斯随机策略 Actor
│   ├── grpo_optimizer.py      #   GRPO 策略优化（真实 log prob）
│   ├── sft_finetune.py        #   SFT 专家微调
│   ├── config.py              #   超参数配置
│   └── checkpoints/           #   训练产物（metrics.json）
├── 02_3DCNN/                   # 模块二：3D-CNN 智能监测
│   ├── pointcloud_pipeline.py #   点云预处理 + 五种物理变化模拟
│   ├── model.py               #   3D U-Net + skip connections
│   ├── change_detection.py    #   双时相变化检测（5类）
│   ├── trend_prediction.py    #   时序劣化趋势预测（LSTM）
│   ├── config.py
│   └── checkpoints/           #   训练产物（change_detector.pth）
├── 03_narrative_engine/        # 模块三：多模态叙事引擎
│   ├── kg_builder.py          #   知识图谱构建（实体-关系-三元组）
│   ├── user_profiling.py      #   用户画像推断（LSTM + 规则）
│   ├── rag_generator.py       #   RAG 内容生成（千人千面）
│   ├── validator.py           #   历史正确性校验器（v2 三层校验）
│   ├── traceability.py        #   一键溯源
│   └── config.py
├── index.html                  # 数字展厅（44件井冈山文物 3D）
├── DATA_POLICY.md              # ⚠️ 数据政策说明（必读）
├── README.md
└── .gitignore
```

## 🔬 技术亮点（v2 改进）

| 模块 | v1 问题 | v2 修复 |
|------|---------|---------|
| RLVR | log prob 用 -MSE 近似，GRPO ratio 恒≈1 | 高斯随机策略，真实 log prob，GRPO 有效更新 |
| 3D-CNN | 训练标签与物理变化不对应 | 5 种变化类型各对应独立物理模拟函数 |
| 叙事引擎 | 校验器为空壳 | 三层真校验：属性一致性 + 时代错误 + 不确定词黑名单 |

## ⚠️ 数据说明

**本仓库不含真实文物数据。** 原因见 [DATA_POLICY.md](DATA_POLICY.md)（博物馆保密协议）。

- 开发和测试使用 `SyntheticDataProvider`（基于物理规则的模拟数据）
- 真实数据接入通过实现 `DataProvider` 抽象接口即可，无需修改算法代码
- 训练证据保留在 `checkpoints/` 目录

## 环境要求

```bash
# Python 3.10+
pip install torch numpy  # 核心依赖
# 叙事引擎额外依赖（可选）
pip install scikit-learn
```

## 运行

```bash
# 模块一：RLVR 修复训练
python 01_RLVR/rlvr_train.py

# 模块二：3D-CNN 变化检测
python 02_3DCNN/change_detection.py

# 模块三：叙事引擎演示
python 03_narrative_engine/rag_generator.py

# 使用统一数据接口
python -c "from common import get_data_provider; p = get_data_provider(); print(p.get_artifact_scan('JGS-001'))"
```

## 📊 训练证据

| 文件 | 说明 |
|------|------|
| `01_RLVR/checkpoints/metrics.json` | RLVR 训练全周期指标 |
| `02_3DCNN/checkpoints/change_detector.pth` | 3D-CNN 变化检测模型权重 |
| 运行各模块 `demo()` 函数 | 实时输出训练/评估结果 |
