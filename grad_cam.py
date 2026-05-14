import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        probabilities = torch.softmax(logits, dim=1)
        confidence = float(probabilities[:, class_index].item())
        return cam, class_index, confidence

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image


def overlay_heatmap(image, heatmap, alpha=0.45):
    image_array = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.float32) / 255.0
    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = (1 - alpha) * image_array + alpha * colored_heatmap
    return np.clip(overlay, 0, 1)


def save_result(original, heatmap, overlay, label, confidence, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"{label}\nconfidence: {confidence:.2%}")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU-friendly Grad-CAM kata using MobileNetV2.")
    parser.add_argument(
        "--image",
        default="sample_images/idea-wall-short-v2.png",
        help="Path to an input image.",
    )
    parser.add_argument(
        "--output",
        default="outputs/grad_cam_result.png",
        help="Where to save the visualization.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help="Optional ImageNet class index. Defaults to the model's top prediction.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity in the overlay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = script_dir / image_path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()

    original_image = load_image(image_path)
    preprocess = weights.transforms()
    image_tensor = preprocess(original_image).unsqueeze(0)

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    try:
        heatmap, class_index, confidence = grad_cam(image_tensor, args.class_index)
    finally:
        grad_cam.close()

    label = weights.meta["categories"][class_index]
    overlay = overlay_heatmap(original_image, heatmap, alpha=args.alpha)
    resized_original = original_image.resize((heatmap.shape[1], heatmap.shape[0]))
    save_result(resized_original, heatmap, overlay, label, confidence, output_path)

    print(f"Predicted class: {class_index} - {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
