# Grad-CAM Code Kata

This kata shows how Grad-CAM can explain which image regions most influenced a CNN classifier prediction.

It is designed to run on CPU. The first run downloads MobileNetV2 ImageNet weights through `torchvision`; later runs reuse the local cache.

## Python Version

This project was originally created with Python 3.12.4. If you are using Python 3.10, install the pinned Python 3.10 requirements instead of the default unpinned file:

```bash
pip install -r requirements-py310.txt
```

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
pip install -r requirements-py310.txt
python -m ipykernel install --user --name grad-cam-code-kata --display-name "Python (grad-cam-code-kata)"
```

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
