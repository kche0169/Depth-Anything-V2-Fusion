# Radar–RGB 深度融合（实现说明、使用与注意事项）

本文档汇总了我在仓库中为支持毫米波雷达（RadarRGBD）与 RGB/深度融合所做的代码改动、你需要完成的数据准备 / 训练步骤、以及详细的注意事项与调试建议。目标是让你能复现、微调和对比三种融合策略：

- ConcatFusion（默认：concat + 1x1 proj + conv + gate，轻量）
- SEFusion（Squeeze-and-Excitation 样的通道加权，超轻量）
- CrossAttentionFusion（跨模态注意力，表达力强但代价高）

我已把三种融合算子和基本的 radar 数据加载 / 投影实现到仓库，下面逐节说明。

## 目录
- 概览
- 我已实现的改动（文件清单与说明）
- 你需要做/准备的工作
- 如何运行 smoke test（快速验证）
- 如何做带 radar 的推理（run.py）
- 如何训练三个融合模型（示例命令与超参建议）
- 数据格式与投影细节（依据 RadarRGBD / show_fig.py）
- 三种融合方法详解与实现细节
- 已知限制 / 注意事项 / 调试策略
- 下一步建议与可选扩展

---

## 概览

实现目标：在 Depth-Anything-V2 的基础上加入雷达支持与可切换的融合算子，使你能在同一代码库中对三种融合策略进行对比实验（同模型/相同超参下）。

实现方式：
- 在 `depth_anything_v2/` 下新增 `radar.py`（Radar loader + RadarEncoder）与 `fusion.py`（三种 Fusion 实现）；
- 修改 `depth_anything_v2/dpt.py` 以支持可选 fusion（通过 `fusion_type` 参数在模型构造时启用）；
- 修改 `run.py` 新增 `--radar-dir` 参数以支持 demo 推理时加载 radar `.mat`；
- 新增 `tools/test_fusion.py` 做 smoke test。

实现原则：
- 后向兼容：当没有提供 radar 时模型行为不变；
- 轻量优先：先实现低开销融合（Concat / SE），Cross-Attention 放在低分辨率层以控制开销；
- 可配置：通过 `fusion_type` 参数切换融合算子。

---

## 我已实现的改动（文件清单与说明）

新增文件：
- `depth_anything_v2/radar.py`
  - `load_radar_mat(mat_path, ...)`：读取 MATLAB `.mat`（字段 `XYZ`），使用 `parameters/Camera_Intrinsics.mat` 与 `parameters/Radar2RGB_Extrinsics.mat` 把雷达点云投影到像素平面，返回稀疏深度图（HxW, float32）与二值 mask（HxW）。
  - `RadarEncoder`：轻量 CNN，多尺度输出，用于将稀疏深度/雷达热图编码为多尺度特征供融合使用。

- `depth_anything_v2/fusion.py`
  - `ConcatFusion`：concat (feat_v, feat_r) -> 1x1 proj -> 3x3 conv -> gate（sigmoid），输出与视觉特征按 gate 混合。
  - `SEFusion`：concat -> 全局池化 -> MLP -> sigmoid -> 通道加权 -> 1x1 投影回视觉通道（Squeeze-and-Excitation 样式）。
  - `CrossAttentionFusion`：视觉 query，雷达 key/value 的轻量 cross-attention（在雷达点数少或低分辨率层适用）。

修改文件：
- `depth_anything_v2/dpt.py`：
  - 导入 `radar` 与 `fusion` 模块；
  - 在 `DepthAnythingV2.__init__` 中新增可选 `fusion_type` 与 `radar_in_ch` 参数（当 `fusion_type` 非 None 时会构建 `RadarEncoder` 与 `self.fusions`）；
  - 扩展 `forward(self, x, radar=None)` 支持输入 `radar` 张量（Bx1xH xW）；若提供则通过 `RadarEncoder` 产生 `radar_feats` 并在 `DPTHead.forward` 中尝试在转换为空间特征后调用 fusion 模块。
  - 扩展 `infer_image(..., radar_path=None)`：可传入 `.mat` 路径以做 demo 推理。

- `run.py`：新增 CLI 参数 `--radar-dir`，尝试按图片文件名 stem 匹配 radar `.mat` 文件并把找到的文件路径传给 `infer_image`。

新增：
- `tools/test_fusion.py`：最小 smoke-test（默认不使用 radar），用于验证模型前向/推理路径能运行。

改动摘要（代码位置提醒）：
- DPTHead：插入 fusion 调用点（在 transformer tokens 转为空间特征后、`projects[i]` 前）。
- DepthAnythingV2：新增 fusion config 支持与 radar encoder 构建。

---

## 你需要做 / 准备的工作

1. 环境（强烈建议使用 conda/mamba）：
   - 推荐创建隔离环境（避免在 root/global 环境中频繁 upgrade/uninstall）：
     ```bash
     conda create -n dav2 python=3.10 -y
     conda activate dav2
     conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
     pip install -r requirements.txt
     ```
   - 如果你仅想快速 smoke-test 且不使用 GPU，可安装 CPU 版 torch：
     ```bash
     python -m pip install "torch==2.1.2+cpu" torchvision --index-url https://download.pytorch.org/whl/cpu
     ```

2. 数据准备（必须）：
   - 确保 RadarRGBD 数据集的 `.mat` 文件在某目录（例如 `data/radar_mat/`），且每帧 `.mat` 文件的文件名 stem（例如 `00001.mat`）能对应到同名的 RGB/ depth 文件（`00001.jpg` / `00001.png`）。
   - 确保仓库中存在相机参数：
     - `parameters/Camera_Intrinsics.mat`
     - `parameters/Radar2RGB_Extrinsics.mat`
     （这是 `show_fig.py` 使用的路径；`load_radar_mat` 默认使用相同路径）

3. 如果要训练：准备用于训练/验证的 split（建议：70/20/10），并确保 dataloader 返回 `(image, radar, depth_target)`，且在 augmentation/resize 时对 image 与 radar 做相同的几何变换保持对齐。

4. 若要做 Cross-Attention：确认显存足够或限制 attention 在低分辨率层，并对雷达点数做筛选（top-K / voxel 下采样）。

---

## 如何运行 smoke test（快速验证）

目的：验证代码导入、模型前向和推理路径正常（不依赖 radar 或 GPU）。

在项目根目录运行：

```bash
python tools/test_fusion.py
```

期望输出：打印 depth 的 shape（例如 `(H, W)`），并且无 ImportError / RuntimeError。如果出现 `ModuleNotFoundError: torch`，请检查并激活你正确的 conda 环境（见上面环境步骤）。

---

## 如何做带 radar 的推理（run.py）

示例：假设 radar `.mat` 文件放在 `data/radar_mat` 下，且每个 `.mat` 的文件名与对应 `assets/examples/demo01.jpg` 的 stem 一致（`demo01.mat`）：

```bash
python run.py --encoder vits --img-path assets/examples --outdir depth_vis --radar-dir data/radar_mat
```

行为：
- `run.py` 会为每张图片计算文件名 stem（`demo01`），查找 `--radar-dir/demo01.mat`。若存在，会把该 mat 投影到模型输入尺寸并传入 `infer_image(..., radar_path=...)`；若不存在则按单模态推理。

注意：`load_radar_mat` 默认把投影结果 resize 到 `input_size`（`run.py` 的 `--input-size`），并在模型内部将 radar 张量传给 `RadarEncoder`。

---

## 如何训练三种融合模型（示例命令与超参建议）

（提示：目前我还没修改 `metric_depth/train.py` 的训练循环来完整支持 radar-consistency loss；后续我可以为你添加。下面给出预期的 CLI 格式与超参建议。）

建议训练流程：

1) 训练 ConcatFusion（默认，轻量）

```bash
python metric_depth/train.py \
  --fusion-type concat \
  --radar-dir data/radar_mat \
  --radar-consistency-weight 0.5 \
  --epochs 40 \
  --batch-size 8 \
  --outdir outputs/fusion_concat
```

2) 训练 SEFusion（超轻量）

```bash
python metric_depth/train.py --fusion-type se --radar-dir data/radar_mat --radar-consistency-weight 0.5 --epochs 40 --outdir outputs/fusion_se
```

3) 训练 CrossAttentionFusion（较慢/显存高）

```bash
python metric_depth/train.py --fusion-type cross --radar-dir data/radar_mat --radar-consistency-weight 0.5 --epochs 40 --batch-size 4 --lr 5e-5 --outdir outputs/fusion_cross
```

超参建议（参考）：
- Optimizer: AdamW
- Base lr: 1e-4（Concat/SE），5e-5（Cross）
- Batch size: 8–12（24GB 卡建议 8–12），Cross 时减半
- Epochs: 40–80
- Radar-consistency-weight (λ_r): 0.5 初始尝试（可在 0.1–1.0 范围调整）

评估指标：使用仓库已有的 metric（AbsRel / RMSE / δ1/δ2/δ3），并额外报告 radar 覆盖区域（mask 内像素）的误差。

---

## 数据格式与投影细节（依据 RadarRGBD / show_fig.py）

1. `.mat` 文件格式：
   - 字段 `XYZ`：形状为 (3, N)，每列为雷达点在雷达坐标系下的 (x, y, z)。
   - 每个 timestamp 对应一个 `.mat` 文件，与 RGB/深度图一一对应。

2. 投影步骤（`load_radar_mat` 已实现）：
   - 使用 `Radar2RGB_Extrinsics`（3x3 R 与 3x1 t）把雷达点从 radar frame 转到相机坐标系： `pts_cam = R @ pts + t`。
   - 使用 `Camera_Intrinsics` 投影到像素平面： `uv1 = K @ pts_cam`；`u = uv1[0]/uv1[2]`，`v = uv1[1]/uv1[2]`。
   - 取 `z = pts_cam[2]` 作为深度值（单位 m）。
   - 在像素 (round(u), round(v)) 处赋值深度，若有多个点映射到同一像素，取最小 z（离相机最近）。
   - 生成 `depth_map`（float32, HxW）和 `mask`（uint8, HxW）。若训练使用 `--input-size`，可在 loader 中把 `depth_map` resize 到该尺寸（或在 `load_radar_mat` 中直接传入 `resize_to=(H,W)`）。

3. 数据增强注意：任何几何增强（crop/flip/resize）必须同时对 image 与 radar 投影后的稀疏 map 做相同变换，保证像素级对齐。

---

## 三种融合方法详解（工程实现摘要）

### ConcatFusion（默认）
- 实现位置：`depth_anything_v2/fusion.py` 中 `ConcatFusion`
- 步骤：将视觉特征和雷达特征在通道维拼接 -> 1x1 投影匹配通道 -> 3x3 conv 提取空间信息 -> Gate（sigmoid）动态加权 -> out = conv_out * gate + feat_v * (1-gate)。
- 优点：轻量、易实现、适用性广。

### SEFusion
- 实现位置：`depth_anything_v2/fusion.py` 中 `SEFusion`
- 步骤：拼接通道 -> 全局池化 -> MLP -> sigmoid 得到通道权重 -> 逐通道乘以权重 -> 1x1 投影回视觉通道。
- 优点：极低计算开销，适合作为 baseline 或在极端资源受限时使用。

### CrossAttentionFusion
- 实现位置：`depth_anything_v2/fusion.py` 中 `CrossAttentionFusion`
- 思路：视觉 features 做 query，雷达 features 做 key/value，计算 attention 后把 context 投影回视觉空间并做残差相加。
- 注意：为了降低复杂度，代码在低分辨率层或当雷达 token 数小于视觉 token 时采用；若雷达点非常密集，需要下采样/筛选。

---

## 已知限制 / 注意事项 / 调试策略（重要）

1. 环境与依赖：
   - 之前在此容器中安装 xformers 曾导致 torch 被升级到不兼容版本；如果你遇到 `ModuleNotFoundError: No module named 'torch'` 或 torchvision/torch 不匹配的问题，请使用独立 conda 环境按本说明安装合适的 torch+torchvision（CUDA 11.8）。

2. 路径与参数文件：
   - `load_radar_mat` 默认读取 `parameters/Camera_Intrinsics.mat` 和 `parameters/Radar2RGB_Extrinsics.mat`；若你的参数文件放在别处，请修改 `load_radar_mat` 调用或把文件复制到该路径。

3. 投影边界问题：
   - 投影点可能落在像素边界外（u<0 或 v<0 或超出图像），这些点会被丢弃。
   - 多点映射到同一像素时，我的实现会取最近（min z），这通常有利于深度估计，但你可以改为 max/均值/置信加权。

4. Radar 点稀疏性与数据增强：
   - 雷达点稀疏且分布不均，在做数据增强（crop/resize）时要确保对 radar depth map 一致变换；训练时可结合稀疏卷积/插值增强或对稀疏点做小范围高斯 Splat 以缓解极端稀疏。

5. CrossAttention 的内存与速度：
   - 在高分辨率层做全局 attention 会内存暴涨；务必把 attention 层限制在低分辨率层（例如 DPT head 的第 2-3 个 stage）或用稀疏 attention 变体。

6. 兼容性：
   - 当 `radar=None` 或没有找到 `.mat` 时，模型会回退到原始行为（单模态 RGB 推理）。

7. 日志与复现：
   - 请确保每次训练的超参（fusion_type、lr、radar-consistency-weight、batch size）记录到配置文件（JSON/YAML）或输出目录下的 `config.json`，便于对比。

8. Smoke Test 出错排查：
   - ImportError / ModuleNotFoundError：检查当前 Python 解释器与 pip 安装是否一致（`python -m pip --version` 与 `which python`）；建议使用 conda 环境并在激活后运行。 
   - 如果 `load_radar_mat` 报错：检查 `.mat` 是否含 `XYZ` 字段与参数 mat 的路径。

---

## 下一步建议与可选扩展

1. 我可以继续为你完成以下工作（选择其一）：
   - 将 `metric_depth/train.py` 修改为原生支持 `--fusion-type`、`--radar-dir`，并实现 radar-consistency loss（并提交 patch 并做小规模训练测试）。
   - 为 Cross-Attention 引入更高效的实现（例如把雷达点作为 K/V、视觉作为 Q，并实现 top-K 雷达点筛选或窗口注意力以降低复杂度）。
   - 实现更好的 radar->image 投影（使用 sub-pixel splat、高斯化 splat、depth interpolation）以改善稀疏点对模型训练的效果。

2. 实验建议（科学对比）：
   - 对比三种 fusion 时固定所有其它超参（lr、epoch、batch 等），只替换 fusion_type；
   - 分别评估全图误差与 radar-mask 内误差，以衡量 radar 对预测的局部贡献；
   - 若数据允许，可进行 ablation：只在 stage2/只在 stage3/两处同时融合，看看融合层级对结果的影响。

---

如果你希望我现在继续：我可以（A）把训练脚本里加入 `--fusion-type` 和 radar-consistency loss 并提交 patch，或（B）先在当前容器跑一次带 radar 的 smoke test（需要你指明 radar 文件夹路径），或者（C）把 Cross-Attention 做成更高效的 top-K 版本。请告诉我你选哪个（或让我立即开始）。

欢迎把任何运行时报错、训练日志或想做的实验告诉我，我会帮你调整和迭代。

---

文件更新摘要（快速回顾）：

- 新增： `depth_anything_v2/radar.py`, `depth_anything_v2/fusion.py`, `tools/test_fusion.py`, `docs/FUSION.md`（本文件）
- 修改： `depth_anything_v2/dpt.py`, `run.py`

最后：如需我现在执行（在容器里）某些操作（如运行 smoke test 或实现训练脚本改动），请回复我你选择的操作（例如 “运行 smoke test” 或 “实现训练脚本”）。

## 扩展方向（研究与工程）

下面列出可持续扩展和优化的方向，按实现难度与预期收益分组，并在每项后标注建议的实现位置、调试要点与优先级（高/中/低）。你可以按优先级逐步迭代，我也可以代劳实现并做 smoke test 与小样本训练验证。

1) 多层/多阶段融合（优先级：高）
   - 意图：在 DPT 的多个 feature stage（例如 stage2、stage3）同时进行融合，提升不同尺度对雷达信息的利用。
   - 实现位置：`depth_anything_v2/dpt.py`（在构造函数创建 `self.fusions` 列表，forward 中在每个 stage 后调用）。
   - 要点：允许每个 stage 有独立的 `fusion_type` 或共享同一类型；最好可配置每层的权重或学习一个融合系数。
   - 调试：先只启用第二层，确认数值稳定，再同时启用第三层，比较 per-mask/whole-image 指标变化。

2) Radar 预处理（splat / UNet / confidence）——输入侧增强（优先级：高）
   - 意图：把稀疏点云转成更稠密、更平滑且带置信度的 map，提高后续融合效果。
   - 实现位置：`depth_anything_v2/radar.py` 新增 `RadarPreprocessor`（可选 learnable UNet 或 deterministic Gaussian splat）。
   - 要点：输出 dense depth map + confidence map，可把 confidence 作为额外通道送入 RadarEncoder 或 fusion block。
   - 调试：用合成点云验证 splat 后的覆盖率与误差；训练时观察 radar-consistency loss 在 mask 区的变化。

3) Cross-Attention 优化（top-K / windowed /低秩）——适用于表达力需求高但要控显存（优先级：中）
   - 意图：保留 Cross-Attention 的表达能力，同时控制时间/内存开销。
   - 实现位置：`depth_anything_v2/fusion.py` 中的 `CrossAttentionFusion` 增加 `topk`、`window_size` 参数与简单的 token 筛选工具。
   - 要点：投影后按置信度/深度/覆盖度挑 top-K 点，或把视觉空间分块（window）只与对应窗口内的 radar token 做 attention。
   - 调试：先在 CPU 或 small GPU 上做单批测试，监测 peak memory 与时间，选择合适的 top-K。

4) 点云原生编码（PointNet / SparseConv / Minkowski）——替代 map-based encoder（优先级：中/低，依赖库）
   - 意图：直接对原始点云提取几何特征，常在点稀疏场景下优于 image-space splat。
   - 实现位置：新增 `depth_anything_v2/radar_pointnet.py` 或在 `radar.py` 增加 `use_pointnet=True` 分支。
   - 要点：PointNet 实现成本低，MinkowskiEngine 的 SparseConv 在稀疏点较多时更高效但需额外依赖。
   - 调试：验证点云编码输出 shape 与可微分性，注意 batch 化点云（不同点数需 pad 或使用拆分/索引）。

5) 时序/多帧融合（Temporal）——利用连续帧稳定性（优先级：中）
   - 意图：融合多帧 radar/visual 特征以提高鲁棒性与覆盖。
   - 实现位置：`depth_anything_v2/radar.py` 或 `dpt.py` 中增加 `TemporalEncoder`（ConvLSTM / TemporalConv / Temporal Attention）。
   - 要点：需要数据按时间对齐并保证 augment 时同帧变换一致；训练复杂度与显存会增加。
   - 调试：做短序列（N=3）实验，观察抖动/噪声是否降低。

6) 损失函数扩展（radar-consistency / uncertainty / mask-weight） —— 改进训练目标（优先级：高）
   - 意图：明确把雷达测量作为强监督（局部）以改善雷达覆盖区域的预测。
   - 实现位置：`metric_depth/train.py` 增加 `--radar-consistency-weight`，计算在 radar mask 内的 MSE（或 inverse-depth MSE）；可选预测不确定性并做不确定性加权。
   - 要点：loss 权重要做超参搜索；建议先冻结部分 backbone 再微调融合模块以稳定训练。
   - 调试：记录 mask 内与全图误差曲线，确保 radar-loss 不至于主导整个训练（导致全图退化）。

7) 轻量化与部署（AMP / 量化 / 剪枝 / ONNX）——落地工程（优先级：中）
   - 意图：在保证精度的前提下降低推理延迟与模型大小，利于部署到边缘设备。
   - 实现位置：训练/导出脚本（`tools/export_*` 或 `metric_depth/` 下新增导出脚本）。
   - 要点：使用 AMP（自动混合精度）训练，导出时尝试 post-training quantization 或 QAT；使用 TorchScript / ONNX 并在目标平台上验证精度。

8) 实验管理与自动化（优先级：高/工具化）
   - 意图：保证实验可复现，便于大规模对比。
   - 实现位置：新增 `tools/exp_runner.py`（统一训练/评估三种 fusion），保存 `config.json`，并输出 per-mask 与全图指标。
   - 要点：固定 seed、TensorBoard/W&B 日志、保存最佳模型与配置，便于比较与复现实验结果。

9) 单元测试与 CI（优先级：中）
   - 意图：降低回归风险，确保基础函数（radar loader、fusion blocks）在重构后仍然正确。
   - 实现位置：`tools/tests/test_radar_loader.py`, `tools/tests/test_fusion_blocks.py`。
   - 要点：写快速的合成数据测试（少量点云、随机特征），作为 CI 的 smoke-tests。

10) 研究方向（可选、长期）
   - 不确定性建模：同时预测深度与不确定性，并用不确定性引导融合权重。
   - 自监督 / 半监督方法：在雷达稀疏区域外使用自监督几何一致性或对比学习增强训练数据利用率。
   - 多任务联合：把语义/边缘检测与深度估计联合训练，以利用语义边界改善深度上采样。

---

如何开始（建议的第一步小计划）
- 第 1 天：实现并集成 `RadarPreprocessor`（Gaussian splat 或小 UNet）与 radar-consistency loss（在 `metric_depth/train.py` 中）——收益大且实现成本中等。
- 第 2 天：把 `ConcatFusion` 扩展为 multi-layer residual block，并在 stage2/stage3 启用多层融合，跑一次短周期（2–5 epoch）微调实验。
- 第 3 天：如需要 Cross-Attention，则实现 top-K 筛选并评估显存/时间开销，决定是否继续优化为窗口化 attention 或引入 efficient attention lib（xformers/flash-attn）。

如果你同意我按上述“小计划”推进，我可以立刻开始实现第 1 项（在训练脚本中加入 `--radar-consistency-weight` 并实现 `RadarPreprocessor`），随后做小样本训练/烟雾测试并把结果与代码 patch 提交到仓库。

