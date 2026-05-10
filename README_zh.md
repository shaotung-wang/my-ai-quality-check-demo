# my-ai-quality-check-demo（中文说明）

本仓库已重新组织为更清晰的包结构：核心功能集中到 `app/core/` 下，便于阅读、维护与后续打包部署。

注意：仓库中的数据集（`datasets/metal_defects/`）与模型权重（如 `runs/.../weights/*.pt`、`yolov8n.pt` 等）均被保留，未被删除。

## 项目概况（简要）

- 目标：为小型金属轴加工厂实现工业视觉质检的 Demo，支持多相机输入、按“轴”聚合判定并保证低漏检。
- 当前实现：核心模块已搬到 `app/core/`，并保留占位的 EfficientAD 推理（默认返回安全的 0.0 分数），方便你日后替换为真实推理模型。

## 主要文件与目录（关键项）

```
my-ai-quality-check-demo/
├─ app/
│  ├─ core/
│  │  ├─ config.py           # 核心配置
│  │  ├─ inference.py        # EfficientAD 推理占位（加载真实模型后替换）
│  │  ├─ aggregator.py       # 轴级聚合逻辑
│  │  ├─ workers.py          # 摄像头 + 推理 worker（线程）
│  │  ├─ main_window.py      # GUI 主窗口实现（PySide6）
│  │  └─ pipeline_demo.py    # 离线模拟与演示
│  ├─ entry.py               # GUI 程序入口
│  └─ (包装模块 app.inference/app.workers/app.gui/app.tools)
├─ datasets/metal_defects/   # 保留的数据集
├─ runs/                     # 训练输出 / 权重（保留）
├─ main.py                   # 启动脚本（委托给 app.entry）
├─ train_model.py            # 训练说明 / 占位脚本（EfficientAD 方向）
├─ requirements.txt          # 依赖清单
├─ README.md                 # 英文/更新版说明（已存在）
└─ README_zh.md              # 本文件（中文说明）
```

## 快速开始（离线演示）

1. 建议在虚拟环境中安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 运行离线模拟（使用 `datasets/metal_defects/train/images` 下的图片）：

```bash
python3 -c "from app.core.pipeline_demo import simulate_run; simulate_run(num_shafts=10, frames_per_shaft=1)"
```

说明：当前 `app/core/inference.py` 为占位实现（若未配置 EfficientAD 的运行时模型，推理分数为 0.0）。当你导出 EfficientAD 的推理模型并配置路径后（见下文），即可获得真实异常检测结果。

## 启动 GUI（需图形环境与 PySide6）

```bash
python3 main.py
```

GUI 将调用 `app.entry.main()` 并使用 `app.core.main_window.MainWindow`。

## 如何准备与接入 EfficientAD（建议）

1. 训练：在服务器或工作站上训练 EfficientAD（或你选择的异常检测模型），训练数据以大量良品（non-defective）为主；少量缺陷图可用于阈值调优与验证。
2. 导出：将训练后的模型导出为推理友好的格式（推荐 TorchScript 或 ONNX）。
3. 放置与配置：将导出的模型文件放到仓库中某个路径，并在 `app/core/config.py` 中修改 `EFFICIENTAD_MODEL_PATH` 指向该文件。
4. 集成：我可以替你把 `app/core/inference.py` 中的占位加载替换为真实的 ONNX/TorchScript 加载 + 推理（使用 `onnxruntime` 或 PyTorch 的 TorchScript + MPS），并做一次性能基准测试（在 M2 Mac mini 上测延迟/吞吐）。

## 设计与注意事项摘要

- 聚合策略：`app/core/aggregator.py` 支持多种聚合策略（`max`、`topk_mean`、`quantile`），可在 `app/core/config.py` 中配置 `AGG_*` 参数。
- 轴级判定：推荐基于多帧（整个旋转周期）取 `max` 或高分位数来做最终判定，保证对“无瑕疵的工件必须确实无瑕疵”的要求可通过保守阈值调整实现（使用已知缺陷样本校准阈值）。
- 性能：目标是在 M2 Mac mini 上尽量实现 ≤1s 的判定延迟。达成手段包括：缩小输入到 512、批量推理、使用 onnxruntime/Metal EP 或 PyTorch MPS、以及缓冲+按轴批判定策略。

## 后续可选工作（我可以帮你实现）

- 把占位的推理替换为 ONNX/TorchScript 的真实加载（并在 M2 上做性能测试）。
- 将 `Aggregator` 集成进 `workers` 做实时轴级判定（含可选的视觉 shaft-id 映射模块）。
- 增加阈值调参脚本与线上记录（便于持续优化）。
- 把仓库整理为可打包发布的 Python 包（pyproject / pip installable）。

如果你想让我继续实现其中某一项，请直接告诉我我会开始实施，并在每一步之后运行必要的测试与报告结果。

