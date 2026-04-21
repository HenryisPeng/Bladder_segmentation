# 膀胱超声图像分割

这是一个基于 CNN U-Net 的膀胱超声图像分割项目，适配当前数据目录：

```text
data/
  bladdercancer/
    train/
    val/
    test/
  bladder_masks/
    train/
    val/
    test/
  bladder_masks_auto/
    train/
    val/
    test/
```

## 重要说明

当前 `data/bladdercancer` 目录中只有原始图像，没有发现任何分割标注文件，因此：

1. 可以直接运行推理脚本，但模型需要先训练或加载你已有的权重。
2. 如果要训练监督式分割模型，必须补充分割掩膜。
3. 掩膜默认目录结构如下：

```text
data/bladder_masks/
  train/
  val/
  test/
```

每张掩膜文件名需要与原图同名，例如：

```text
data/bladdercancer/train/10.jpg
data/bladder_masks/train/10.png
```

掩膜应为二值图：

- 背景像素值：`0`
- 膀胱区域像素值：`255`

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 先创建掩膜目录骨架

```bash
python3 scripts/create_mask_scaffold.py \
  --image-root data/bladdercancer \
  --mask-root data/bladder_masks
```

这个脚本会：

- 创建 `train/val/test` 对应的 mask 目录
- 输出每个 split 需要标注的文件数量

## 自动生成粗掩膜

如果你现在还没有人工标注，可以先生成一版粗掩膜，再人工修正：

```bash
python3 scripts/generate_coarse_masks.py \
  --image-root data/bladdercancer \
  --output-root data/bladder_masks_auto \
  --save-overlay
```

这个脚本会：

- 为 `train/val/test` 批量生成粗分割 mask
- 把 mask 保存到 `data/bladder_masks_auto/train|val|test`
- 把带轮廓的检查图保存到 `data/bladder_masks_auto/train_overlays|val_overlays|test_overlays`

注意：

- 这只是基于传统图像处理生成的伪标签，不是高质量金标准标注
- 建议先人工检查和修正，再拿去训练
- 如果你要直接用这些粗 mask 训练，把 `--mask-root` 指向 `data/bladder_masks_auto`

## 训练模型

```bash
python3 train.py \
  --image-root data/bladdercancer \
  --mask-root data/bladder_masks \
  --epochs 80 \
  --batch-size 8 \
  --image-size 256 \
  --learning-rate 1e-3
```

训练输出会保存在：

```text
outputs/
  checkpoints/
  plots/
```

## 对 test 集做分割推理

```bash
python3 predict.py \
  --image-root data/bladdercancer/test \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output-dir outputs/predictions/test
```

## 生成带轮廓叠加的可视化图

```bash
python3 predict.py \
  --image-root data/bladdercancer/test \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output-dir outputs/predictions/test \
  --save-overlay
```

## 文件说明

- `train.py`：训练与验证
- `predict.py`：批量推理与可视化
- `scripts/create_mask_scaffold.py`：建立标注目录骨架
- `bladder_segmentation/dataset.py`：数据读取
- `bladder_segmentation/model.py`：U-Net 网络
- `bladder_segmentation/losses.py`：损失函数
- `bladder_segmentation/metrics.py`：Dice / IoU 指标

## 标注建议

如果你还没有 mask，推荐使用任一标注工具手工勾画膀胱区域后导出二值掩膜，例如：

- CVAT
- Labelme
- ITK-SNAP

导出后请统一成与原图同名的 `png` 二值图，再放入 `data/bladder_masks/train`、`data/bladder_masks/val`、`data/bladder_masks/test` 中。
