# Grad-CAM Sharing Session Guide

## Session Goal

Help the team understand how Grad-CAM turns a classifier prediction into a visual debugging signal.

By the end, participants should be able to explain:

- What feature maps and gradients contribute to Grad-CAM.
- Why Grad-CAM is useful for model debugging.
- Why a correct prediction can still be suspicious.
- How Grad-CAM changes when the target class changes.

## Suggested Timing

### 1. Problem Setup - 5 minutes

Ask:

- The model predicts a class, but can we trust why it predicted it?
- What would be suspicious evidence for an image classifier?

Show one input image and the model's predicted label.

### 2. Grad-CAM Concept - 10 minutes

Explain the core pipeline:

1. Run a forward pass and capture final convolution features.
2. Choose a target class score.
3. Run backward pass and capture gradients.
4. Average gradients across spatial dimensions to get channel weights.
5. Combine weighted feature maps into a heatmap.
6. Overlay the heatmap on the image.

### 3. Code Walkthrough - 15 minutes

Open `grad_cam.py` and focus on:

- `register_forward_hook`
- `register_full_backward_hook`
- `weights = gradients.mean(...)`
- `cam = (weights * activations).sum(...)`
- `overlay_heatmap(...)`

Keep the walkthrough centered on data shapes:

- Input tensor: `[1, 3, 224, 224]`
- Activations: `[1, channels, h, w]`
- Gradients: `[1, channels, h, w]`
- Final heatmap: `[224, 224]`

### 4. Hands-On Experiments - 15 minutes

Try:

```bash
python grad_cam.py --image sample_images/idea-wall-short-v2.png
```

Then ask participants to change one thing:

- Use another image.
- Change `--alpha`.
- Force a class with `--class-index`.
- Try an image that is not ImageNet-like, such as a screenshot or diagram.

### 5. Discussion - 10 minutes

Prompts:

- Did the model focus on the expected region?
- Is the top prediction meaningful for this image?
- What kinds of images make Grad-CAM less useful?
- How would you use this in a real model debugging workflow?

## Good Takeaways

- Grad-CAM is a debugging aid, not a proof of correctness.
- Heatmaps are class-specific; changing the target class can change the explanation.
- Out-of-distribution images can produce confident but unhelpful explanations.
- For production, Grad-CAM is most useful when paired with error analysis and representative data.
