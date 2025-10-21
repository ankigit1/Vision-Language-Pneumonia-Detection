"""
main_inference.py
Run inference on a single chest X-ray using the trained Siamese DenseNet121 + LBP + VLM Guidance model.
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.siamese_densenet_lbp import SiameseDenseLBP, compute_lbp_batch, split_left_right
from models.vlm_guidance import VisionLanguageGuidance
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------- CONFIG ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoints/pneumonia_siamese_vlm_lambda0.25.pt"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------- PREPROCESS -------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
    tensor = tfm(img).unsqueeze(0).to(DEVICE)
    return tensor, np.array(img.resize((IMG_SIZE, IMG_SIZE)))

# ---------------------- LOAD MODEL -------------------
def load_model():
    model = SiameseDenseLBP().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    return model

# ---------------------- RUN INFERENCE ----------------
def run_inference(model, img_tensor):
    """Run forward pass and return class probabilities."""
    logits, cos, xl, xr = model(img_tensor)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, float(cos.item()), xl, xr

# ---------------------- CLIP GUIDANCE ----------------
def compute_vlm_guidance(vlm, xl, xr, cos_val):
    """Compute CLIP-based laterality guidance (if available)."""
    if vlm is None or not vlm.ok:
        return None, None, None
    with torch.no_grad():
        import clip
        tL = vlm.model.encode_image(F.interpolate(xl[:, :3], size=224))
        tR = vlm.model.encode_image(F.interpolate(xr[:, :3], size=224))
        tL = F.normalize(tL, dim=-1)
        tR = F.normalize(tR, dim=-1)
        simL = (tL @ F.normalize(vlm.model.encode_text(vlm.prompts["left"]), dim=-1).T).squeeze().item()
        simR = (tR @ F.normalize(vlm.model.encode_text(vlm.prompts["right"]), dim=-1).T).squeeze().item()
        s_star = 1 - abs(simL - simR)
    return simL, simR, s_star

# ---------------------- GRAD-CAM ---------------------
def generate_gradcam(model, img_tensor, pred_idx, img_rgb):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class LogitsOnly(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): out, _, _, _ = self.base(x); return out

    target_layer = [model.encoder[-2]]  # Dense block 4
    cam = GradCAM(model=LogitsOnly(model), target_layers=target_layer)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (IMG_SIZE, IMG_SIZE))
    img_norm = img_rgb / 255.0
    vis = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    save_path = os.path.join(RESULT_DIR, "gradcam_inference.png")
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"📸 Grad-CAM saved at: {save_path}")
    return save_path


# ---------------------- MAIN -------------------------
def main(image_path):
    print(f"\n🔍 Running inference on: {image_path}")
    model = load_model()
    vlm = VisionLanguageGuidance(DEVICE)

    img_tensor, img_rgb = preprocess_image(image_path)
    pred_idx, probs, cos_val, xl, xr = run_inference(model, img_tensor)

    labels = ["NORMAL", "PNEUMONIA"]
    pred_label = labels[pred_idx]

    simL, simR, s_star = compute_vlm_guidance(vlm, xl, xr, cos_val)

    print("\n================ PREDICTION SUMMARY ================")
    print(f"Predicted Class: {pred_label}  (Normal={probs[0]:.3f}, Pneumonia={probs[1]:.3f})")
    print(f"Left–Right Cosine Similarity (model): {cos_val:.3f}")
    if simL is not None:
        print(f"🩻 CLIP Laterality Prompts:")
        print(f"  Left  ('left lung opacity')  → {simL:.4f}")
        print(f"  Right ('right lung opacity') → {simR:.4f}")
        print(f"  Target similarity s* = 1 - |t_L - t_R| = {s_star:.4f}")
    print("====================================================")

    # Generate Grad-CAM
    generate_gradcam(model, img_tensor, pred_idx, img_rgb)

# ---------------------- ENTRY POINT ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference for Pneumonia Detection Model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to chest X-ray image")
    args = parser.parse_args()
    main(args.image_path)
