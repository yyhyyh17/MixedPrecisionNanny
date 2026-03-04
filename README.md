# MixedPrecisionNanny

混合精度训练保姆——深度学习FP16/BF16训练问题诊断工具。

## 项目背景

在深度学习训练过程中，混合精度训练（FP16/BF16）是提高训练效率的重要手段，但往往会面临精度下降（"掉点"）的问题。根据业务影响程度，我们将掉点分为两类：

| 类型 | 描述 | 业务可接受性 |
|------|------|-------------|
| **类型1：轻微掉点** | 由数值精度差异导致的模型精度轻微下降 | ✅ 可接受，可业务使用 |
| **类型2：显著掉点** | 训练崩溃、精度显著下降，不符合预期 | ❌ 不可接受 |

从业务角度出发，我们的目标是：**最大化类型1的发生，最小化甚至消除类型2的发生**。

为此，我们需要一套完整的工具和方法论来预防和诊断类型2的问题。

## 核心功能

MixedPrecisionNanny 旨在通过以下三个核心功能模块，帮助业务团队在混合精度训练前、中、后全链路保障训练稳定性：

### a. 静态代码检查（训练前预防）

在训练开始前，对模型代码进行静态分析，提前识别潜在的数值风险点：

- **溢出敏感算子检测**：识别在 FP16/BF16 下容易发生数值溢出或下溢的算子
- **异常值风险分析**：检测可能导致 NaN/Inf 的运算模式
- **最佳实践建议**：提供针对性的代码改进建议

**典型检测场景**：
- Softmax、LayerNorm、CrossEntropy 等易溢出算子
- 梯度缩放（Gradient Scaling）配置检查
- Loss Scaling 策略合理性验证

### b. 轻量化梯度监控（训练中观测）

在训练过程中，以轻量化的方式实时监控模型状态，捕捉异常信号：

- **梯度统计监控**：实时跟踪梯度范数、梯度分布
- **数值异常告警**：及时检测 NaN/Inf、异常大/小梯度
- **层级粒度追踪**：支持模型各层的细粒度监控
- **低性能开销**：确保监控本身不影响训练效率

### c. 可视化分析界面（训练后诊断）

提供直观的可视化工具，帮助业务人员深入分析训练过程中的数值行为：

- **训练曲线可视化**：Loss、梯度范数、学习率等关键指标
- **数值分布热力图**：各层激活值、权重的数值分布变化
- **异常时刻定位**：快速定位异常发生的时间点和位置
- **对比分析**：支持 FP32 与 FP16/BF16 训练的对比

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   MixedPrecisionNanny 工作流                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   训练前检查  │───▶│   训练中监控  │───▶│   训练后分析  │      │
│  │  (StaticCheck)│    │  (Monitoring) │    │  (Visualizer) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│          │                   │                   │             │
│          ▼                   ▼                   ▼             │
│   ┌────────────┐      ┌────────────┐      ┌────────────┐      │
│   │ 风险算子识别 │      │ 梯度异常告警 │      │ 数值行为分析 │      │
│   │ 代码问题扫描 │      │ 实时统计上报 │      │ 可视化报表   │      │
│   └────────────┘      └────────────┘      └────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
MixedPrecisionNanny/
├── tracer/                  # 采样与 Hook 管理
│   ├── sampler.py           # 采样策略（周期 / 触发密集采样）
│   └── hook_manager.py      # PyTorch forward/backward hook 注册与管理
├── analyzer/                # 数值分析与告警
│   └── numerical_checker.py # 张量统计计算 + 告警规则
├── storage/                 # 数据持久化
│   └── sqlite_writer.py     # 异步 SQLite 写入器（WAL 模式）
├── examples/                # 训练示例
│   ├── train_resnet_classification.py  # ResNet-18 分类训练 + 监控
│   ├── train_yolo_detection.py         # Tiny-YOLO 检测训练 + 监控
│   └── README.md
├── tests/                   # 测试套件（183 个测试）
│   ├── conftest.py          # 公共 fixtures 和工具函数
│   ├── test_sampler.py      # Sampler 单元测试
│   ├── test_numerical_checker.py  # 数值分析单元测试
│   ├── test_sqlite_writer.py      # SQLite 写入器单元测试
│   ├── test_hook_manager.py       # HookManager 单元测试
│   ├── test_nanny.py              # MixedPrecisionNanny 集成测试
│   └── test_integration_training.py  # 端到端训练集成测试
├── nanny.py                 # 主入口：MixedPrecisionNanny 类
├── cli.py                   # 命令行查询工具
├── mcp_server/              # AI Agent 静态代码检查（MCP Server）
│   ├── server.py
│   ├── skills/
│   └── prompts/
├── skills/                  # 混合精度问题知识库
├── docs/                    # 设计文档
├── check_model.py           # 极简静态检查脚本
├── monitor_simple.py        # 极简监控脚本
└── README.md
```

## 目标用户

- **算法工程师**：排查混合精度训练中的精度问题
- **训练平台开发者**：集成监控能力到训练框架
- **业务负责人**：评估混合精度训练的风险与收益

## 环境安装

### 安装 Miniconda（已安装可跳过）

```bash
# macOS Apple Silicon（arm64）
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3

# 初始化 conda（执行一次，之后重开终端生效）
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### 创建项目环境

```bash
# 创建 Python 3.10 环境（只需执行一次）
conda create -n mpnanny python=3.10 -y
conda activate mpnanny

# 安装 PyTorch 和测试依赖（国内用户建议加 -i 镜像源加速）
pip install torch numpy pytest -i https://mirrors.aliyun.com/pypi/simple/
```

### 运行监控

```python
from nanny import MixedPrecisionNanny

nanny = MixedPrecisionNanny(model, trace_interval=100)

for step, (x, y) in enumerate(dataloader):
    with nanny.step(step):
        loss = criterion(model(x), y)
        loss.backward()
    optimizer.step()

nanny.close()
```

### 将精度检测切换为 BF16

默认使用 **FP16** 的数值边界做饱和/下溢检测。若训练使用 **BF16**（`torch.autocast(dtype=torch.bfloat16)`），可改为按 BF16 边界检测：

**方式一：构造时传 `precision`（推荐）**

```python
nanny = MixedPrecisionNanny(model, trace_interval=100, precision="bf16")
```

**方式二：通过 `AlertConfig` 指定**

```python
from analyzer.numerical_checker import AlertConfig

nanny = MixedPrecisionNanny(
    model,
    trace_interval=100,
    alert_config=AlertConfig(precision="bf16"),
)
```

切换为 BF16 后，饱和/下溢阈值将使用 BF16 的数值范围（约 3.4e38 / 1.2e-38），告警文案会显示为 “BF16 saturation / BF16 underflow”。

### 查询监控结果

```bash
# 查看整体摘要
python cli.py summary --db nanny_logs/metrics.db

# 查看所有 ERROR 级告警
python cli.py alerts --db nanny_logs/metrics.db --severity ERROR

# 查看指定 step 的各层统计
python cli.py stats --db nanny_logs/metrics.db --step 100
```

## 训练示例

`examples/` 目录提供了两个完整的训练示例，展示如何在真实模型中集成监控：

| 示例 | 模型 | 任务 | 说明 |
|------|------|------|------|
| `train_resnet_classification.py` | ResNet-18 | 图像分类 | 合成 CIFAR-10 数据，SGD 优化 |
| `train_yolo_detection.py` | Tiny-YOLO | 目标检测 | 合成检测数据，多分支 Loss |

```bash
# ResNet 分类训练
python examples/train_resnet_classification.py

# YOLO 检测训练
python examples/train_yolo_detection.py
```

详见 [examples/README.md](examples/README.md)。

## 快速开始

### 方式1：使用MCP Server（推荐）

通过AI Agent进行自然语言交互式的代码检查：

```bash
# 配置MCP Server到Claude Desktop等客户端
# 详见 mcp_server/README.md
```

### 方式2：使用命令行脚本

```bash
# 静态检查
python check_model.py your_model.py

# 监控训练
from monitor_simple import watch_model
watch_model(model)
```

## 测试

项目共有 **183 个测试**，覆盖各核心模块和端到端训练场景。

### 运行测试

```bash
conda activate mpnanny
cd MixedPrecisionNanny

# 运行所有测试
pytest tests/ -v

# 只跑集成训练测试
pytest tests/test_integration_training.py -v

# 只跑某个模块
pytest tests/test_numerical_checker.py -v
```

### 测试覆盖范围

| 测试文件 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| `test_sampler.py` | 21 | 采样策略：周期模式、触发密集采样、边界情况 |
| `test_numerical_checker.py` | 36 | 张量统计计算、FP16 饱和/下溢检测、各类告警生成 |
| `test_sqlite_writer.py` | 20 | 异步写入、批量提交、WAL 并发读、flush/close 正确性 |
| `test_hook_manager.py` | 29 | Hook 注册/移除、前向/反向数据捕获、层过滤、告警回调 |
| `test_nanny.py` | 22 | 主类集成、step 上下文管理、GradScaler 监控、密集采样触发 |
| `test_integration_training.py` | 55 | 端到端训练场景，验证 Nanny 能检测所有典型问题 |

### 集成测试覆盖的训练问题场景

| 场景 | 触发方式 | 期望告警 |
|------|---------|---------|
| 正常训练 | 标准 MLP | 无告警 |
| FP16 上溢 | 权重 4500，输出 ~72000 > FP16_MAX | `OVERFLOW ERROR` |
| FP16 下溢 | 激活值 × 1e-8，远低于 FP16_MIN_NORMAL | `UNDERFLOW WARNING` |
| NaN 传播 | `log(负数)` | `NAN ERROR` |
| Inf 注入 | 激活值中插入 `inf` | `INF ERROR` |
| 梯度爆炸 | 大权重反向传播，梯度 > 1e4 | `GRAD_EXPLOSION ERROR` |
| 梯度消失 | 极小权重，梯度 < 1e-8 | `GRAD_VANISH WARNING` |
| 大数吃小数 | 大维度 sum，256 × 320 = 81920 > FP16_MAX | `OVERFLOW ERROR` |
| 混合上溢 + NaN | 第一层溢出，第二层产生 NaN | `OVERFLOW + NAN` |
| 渐进式退化 | 权重从正常变为 5000 | 前期无告警，后期 `OVERFLOW` |

### 测试结果（macOS Apple Silicon + PyTorch 2.10）

```
183 passed, 29 warnings in 1.94s
```

## 技术特点

- **MCP协议支持**：与AI Agent自然语言交互
- **自适应阈值**：根据训练数据自动确定告警阈值
- **零侵入监控**：`with nanny.step(step):` 方式接入，改动极少
- **异步存储**：独立线程写 SQLite，不阻塞训练主循环
- **密集采样触发**：检测到 ERROR 级问题后自动切换到密集追踪模式
- **知识驱动**：基于真实案例的问题诊断

## 贡献指南

欢迎提交 Issue 和 PR，共同完善这个工具。

## License

待定
