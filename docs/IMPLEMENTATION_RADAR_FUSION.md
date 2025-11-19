**实现与变更记录 — 雷达模态与可切换融合（由 AI 助手实现）**

更新时间：2025-11-20

该文档详尽记录了我在仓库中为“添加雷达模态支持与可切换融合算子”所做的修改、新增脚本、使用说明与注意事项，便于回溯与复现。

---

1) 概览
- 目标：在 Depth-Anything-V2 基础上引入雷达模态（sparse radar）支持，并使训练脚本可切换融合算子；同时准备数据 split 来映射 image↔depth↔radar，便于训练时计算雷达一致性损失（radar-consistency loss）。
- 我做了三类主要修改：
  - 新增：`tools/generate_splits_with_radar.py`（生成带雷达路径的 split）
  - 修改：`metric_depth/dataset/{hypersim,vkitti2,kitti}.py`（支持第三列 radar_path，返回 `radar_depth`/`radar_mask`）
  - 修改：`metric_depth/train.py`（雷达一致性损失优先使用样本内 radar 张量，失败时回退到已有的 `--radar-dir` 文件匹配逻辑）

2) 新增脚本
- `tools/generate_splits_with_radar.py`（已添加）
  - 作用：读取现有 `dataset/splits/{train,val,test}.txt`（每行为 `image_path depth_path`），在 `dataset/radar/` 下查找匹配的雷达 `.mat` 文件并把第三列写入输出目录 `dataset/splits_with_radar/{train,val,test}.txt`。
  - 匹配策略：先对 `dataset/radar` 建索引（一次遍历），用文件夹名或文件名（无扩展名）与图像文件名的时间戳（到秒）做直接或包含匹配，找到第一个 `.mat` 即视为匹配。找不到则在输出中写 `-` 作为占位。
  - 目的：不要直接覆盖原 split，先写到 `dataset/splits_with_radar` 以便审查与回滚。

  使用示例：
  ```bash
  python3 tools/generate_splits_with_radar.py
  sed -n '1,5p' dataset/splits_with_radar/train.txt
  ```

3) 数据加载器修改（已修改）
- 修改文件：
  - `metric_depth/dataset/hypersim.py`
  - `metric_depth/dataset/vkitti2.py`
  - `metric_depth/dataset/kitti.py`

- 行为变化：
  - 读取 split 时支持 2 列或 3 列（`image_path depth_path [radar_path]`）。
  - 如果第三列存在且非 `-`，加载器会尝试调用 `depth_anything_v2.radar.load_radar_mat(radar_path, resize_to=(H,W))` 来生成稀疏雷达深度图与掩码（depth_map, mask），并在返回的 sample 中附加两个字段：`sample['radar_depth']` (Tensor, shape (1,H,W)) 和 `sample['radar_mask']` (Tensor, shape (1,H,W))。
  - 若 radar 文件缺失或加载失败，加载器会设置 `radar_depth`/`radar_mask` 为 0 张量（shape (1,H,W)），以保证批量拼接（collate）与训练流程稳定。

- 代码要点（加载器行为）：
  - 解析 split 行：`parts = line.split(); img=parts[0]; depth=parts[1]; radar = parts[2] if len(parts)>2 else None`
  - radar 路径支持相对路径或绝对路径：相对路径会按 repo 根解析。

4) 训练脚本改动（已修改）
- 文件：`metric_depth/train.py`
- 主要改动：
  - 新增/已有 CLI：`--fusion-type`、`--radar-dir`、`--radar-consistency-weight`（这些选项允许你开启/配置雷达一致性损失与 fusion 行为）。
  - 雷达一致性损失计算时优先使用样本内的 `radar_depth`/`radar_mask`（来自 split 第三列与加载器），若样本内无有效雷达且 `--radar-dir` 指定，则回退到按 `image_path` 名称尝试在 `--radar-dir` 下匹配 `radar_{ts}.mat` 或 `{ts}.mat` 的逻辑（与之前实现兼容）。
  - 计算细节：构造与 `pred`（网络输出）相同形状的 `radar_depths` 与 `radar_masks`，用掩码求和（mask_sum）归一化后计算 MSE：radar_loss = sum((pred - radar_depths)^2 * mask) / mask_sum，然后乘以权重 `--radar-consistency-weight` 并加到总 loss。

5) 已做的仓库修改（逐文件清单）
- 新增：
  - `tools/generate_splits_with_radar.py` — split 增强脚本（输出到 `dataset/splits_with_radar`）。

- 修改：
  - `metric_depth/dataset/hypersim.py` — 解析第三列 radar 路径，尝试 load 并在 sample 中加入 `radar_depth`/`radar_mask`（缺失时填零张量）。
  - `metric_depth/dataset/vkitti2.py` — 同上。
  - `metric_depth/dataset/kitti.py` — 同上。
  - `metric_depth/train.py` — 优先使用 sample 内 radar；回退到 `--radar-dir` 的匹配；保留 CLI 参数；保证后向兼容。

6) 当前运行状态与注意事项
- `tools/generate_splits_with_radar.py` 已添加，但上次运行时因为手动中断（Ctrl-C）被打断；因此 `dataset/splits_with_radar/` 可能**尚未**完全生成。建议重新运行该脚本以完成索引与写表。
- 数据集目录结构（示例）：
  - `dataset/image/...`
  - `dataset/depth/...`
  - `dataset/radar/...`（仓库内有 `dataset/radar/radar_point2243/...` 形式的目录）
  - 新输出：`dataset/splits_with_radar/{train.txt,val.txt,test.txt}`（每行 `image depth radar_or_-`）

7) 使用与验证步骤（推荐）
- 1) 生成带雷达列的 split（写到 splits_with_radar）：
  ```bash
  python3 tools/generate_splits_with_radar.py
  sed -n '1,6p' dataset/splits_with_radar/train.txt
  ```

- 2) 小样本加载测试（在仓库根执行）：
  ```bash
  python3 - <<'PY'
from metric_depth.dataset.hypersim import Hypersim
ds = Hypersim('dataset/splits_with_radar/train.txt','train', size=(256,256))
for i in range(3):
    s = ds[i]
    print('sample', i, s['image'].shape, s['depth'].shape, s['radar_depth'].shape, int(s['radar_mask'].sum().item()))
PY
  ```

- 3) 运行快速 dry-run（单卡）验证训练+雷达loss：
  ```bash
  # 需要 torchrun/torch.distributed 可用
  torchrun --nproc_per_node=1 metric_depth/train.py --save-path runs/quick --epochs 1 --bs 1 --fusion-type concat --radar-consistency-weight 0.1
  ```

8) 兼容性与设计决策解释
- 为什么把雷达路径写入 split：明确、可靠、便于审核。相比运行时模糊搜索，split 映射最稳健。
- 为何加载器返回零张量而不是 None：避免 DataLoader 的默认 collate 过程失败；统一张量形状让训练 loop 代码更简单（可直接用 sample['radar_mask'] 求和）。
- 索引机制：生成脚本会先遍历 `dataset/radar` 并建立键值索引（folder base / file stem -> 首个 .mat），因此索引耗时受雷达文件数影响，但只需一次。

9) 后续建议（可选）
- 1）完成并审查 `dataset/splits_with_radar`：运行生成脚本直到完成并检查匹配率（多少行带到有效 radar 路径）。
- 2）如果你希望把 splits_with_radar 替换现有 splits（覆盖），先备份并确认无误后再覆盖（脚本当前不覆盖原文件以防不测）。
- 3）若未来想支持非 `.mat` 格式（如 `.npz`/`.h5`），可扩展 `tools/generate_splits_with_radar.py` 的索引过滤与 `depth_anything_v2/radar.py` 的 `load_radar_*` loader。

---

如果你希望，我可以：
- 现在重新运行 `tools/generate_splits_with_radar.py` 直到完成，并把匹配统计（匹配率、示例行）贴给你；或
- 按你的确认把 `dataset/splits_with_radar` 中的内容合并/替换到 `dataset/splits`（需你确认覆盖许可）；或
- 继续做一个小的 dry-run（单卡）来验证训练端的改动。

告诉我下一步要我直接执行哪一项（重新生成 / 覆盖 / dry-run / 其它）。
