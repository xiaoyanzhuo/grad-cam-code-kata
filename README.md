# Grad-CAM Code Kata

This kata shows how Grad-CAM can explain which image regions most influenced a CNN classifier prediction.

It is designed to run on CPU. The first run downloads MobileNetV2 ImageNet weights through `torchvision`; later runs reuse the local cache.

## Python Version

This project was originally created with Python 3.12.4. If you are using Python 3.10, use the CPU-only setup below.

On Linux, installing `torch` from the default PyPI index may download NVIDIA CUDA-related packages. This project does not need them. Use the PyTorch CPU wheel index instead.

## Setup

```bash
cd /Users/xiaoyanzhuo/Documents/grad-cam-code-kata
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name grad-cam-code-kata --display-name "Python (grad-cam-code-kata)"
```

For Python 3.10:

```bash
cd /Users/xiaoyanzhuo/Documents/grad-cam-code-kata
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-py310.txt
python -m ipykernel install --user --name grad-cam-code-kata --display-name "Python (grad-cam-code-kata)"
```

For a CPU-only machine, use the same setup:

```bash
cd /Users/xiaoyanzhuo/Documents/grad-cam-code-kata
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-py310.txt
python -m ipykernel install --user --name grad-cam-code-kata --display-name "Python (grad-cam-code-kata)"
```

The notebook and script run on CPU by default; no GPU or CUDA setup is required.

If Python 3.10 still tries to build a package with `meson-python`, force binary wheels:

```bash
pip install --only-binary=:all: -r requirements-py310.txt
```

The Python 3.10 requirements intentionally do not install `notebook`; they only install the packages needed by the Grad-CAM kernel. You can open Jupyter from an existing Anaconda/Jupyter install and select the `Python (grad-cam-code-kata)` kernel.

## Run

```bash
python grad_cam.py
```

The default command uses `sample_images/idea-wall-short-v2.png` and saves:

```text
grad_cam_kata/outputs/grad_cam_result.png
```

Use your own image:

```bash
python grad_cam.py --image /path/to/image.jpg --output outputs/my_result.png
```

Force Grad-CAM for a specific ImageNet class index:

```bash
python grad_cam.py --image /path/to/image.jpg --class-index 281
```

## Offline Model Checkpoint

By default, `torchvision` downloads the MobileNetV2 ImageNet checkpoint the first time the model is used. If the target machine cannot download it, download the file on another machine:

[mobilenet_v2-7ebf99e0.pth](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth)

Then copy it to the target machine, for example:

```text
model_checkpoints/mobilenet_v2-7ebf99e0.pth
```

The `model_checkpoints/` folder is ignored by git so large checkpoint files are not committed.

Run the script with the local checkpoint:

```bash
python grad_cam.py \
  --weights-path model_checkpoints/mobilenet_v2-7ebf99e0.pth \
  --image sample_images/idea-wall-short-v2.png
```

In the notebook, set:

```python
local_weights_path = notebook_dir / "model_checkpoints" / "mobilenet_v2-7ebf99e0.pth"
```

If `local_weights_path` is `None`, the notebook uses the normal `torchvision` download/cache behavior.

## Using A Different Model

This kata currently uses MobileNetV2:

```python
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
target_layer = model.features[-1]
```

To use another `torchvision` classifier, update four things:

1. Import the model builder and weights enum.
2. Build the model with either `weights=...` or `weights=None` plus `load_state_dict(...)`.
3. Use that model's preprocessing transforms from `weights.transforms()`.
4. Choose the final convolution layer for Grad-CAM.

Example with ResNet18:

```python
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
target_layer = model.layer4[-1]
```

If using a local ResNet18 checkpoint:

```python
model = resnet18(weights=None)
checkpoint = torch.load("model_checkpoints/resnet18-f37072fd.pth", map_location="cpu")
model.load_state_dict(checkpoint)
target_layer = model.layer4[-1]
```

The model checkpoint must match the model architecture. A MobileNetV2 checkpoint cannot be loaded into ResNet18, and a custom fine-tuned model may need a different class-label list.

## Notebook Demo

For a step-by-step sharing session with inline results:

```bash
jupyter notebook grad_cam_kata.ipynb
```

In Jupyter, choose the kernel named `Python (grad-cam-code-kata)`, then run the cells top to bottom. The notebook displays the input image, heatmap, overlay, predicted class, confidence, and saves:

```text
grad_cam_kata/outputs/grad_cam_notebook_result.png
```

## Sharing Flow

1. Run the model and show the predicted ImageNet class.
2. Show the Grad-CAM heatmap.
3. Ask whether the model appears to focus on the expected object.
4. Try an ambiguous or out-of-distribution image.
5. Change `--class-index` and compare how the heatmap moves.

## Key Concepts To Discuss

- Forward hooks capture feature maps from the final convolution block.
- Backward hooks capture gradients for the target class score.
- Channel weights come from global-average-pooled gradients.
- The weighted activation map highlights regions that increased the target score.
- A strong prediction can still rely on surprising or irrelevant image regions.

## Kata Extensions

- Swap MobileNetV2 for ResNet18 and compare heatmaps.
- Run several images and make a small gallery of results.
- Compare top-1 Grad-CAM with Grad-CAM for another class index.
- Add simple augmentations like crop, blur, or brightness changes and watch the heatmap shift.
