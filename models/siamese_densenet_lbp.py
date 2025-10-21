# models/siamese_densenet_lbp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.feature import local_binary_pattern
import numpy as np


# ---------------- Utility: split & LBP ----------------
def split_left_right(x):
    mid = x.shape[-1] // 2
    return x[..., :mid], x[..., mid:]


def compute_lbp_batch(imgs):
    """Compute LBP per image and append as 4th channel."""
    imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()
    lbp_maps = []
    for im in imgs_np:
        gray = (im * 255).mean(axis=2).astype(np.uint8)
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6)
        lbp_maps.append(lbp)
    lbp_tensor = torch.tensor(np.stack(lbp_maps), dtype=torch.float32).unsqueeze(1)
    return torch.cat([imgs, lbp_tensor.to(imgs.device)], dim=1)


# ---------------- Laterality CLIP Loss ----------------
class LateralityCLIPLoss(nn.Module):
    """
    s* = 1 - |t_L - t_R|
    L_lat = (cos_sim(f_L, f_R) - s*)^2
    """

    def __init__(self, device):
        super().__init__()
        try:
            import clip

            self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self.model.eval()
            self.prompt_L = clip.tokenize(["left lung opacity"]).to(device)
            self.prompt_R = clip.tokenize(["right lung opacity"]).to(device)
            self.clip_available = True
        except Exception as e:
            print("❌ CLIP unavailable:", e)
            self.clip_available = False

    @torch.no_grad()
    def get_clip_sim(self, imgs, tokens):
        import clip

        imgs = F.interpolate(imgs[:, :3], size=224)
        img_e = F.normalize(self.model.encode_image(imgs), dim=-1)
        txt_e = F.normalize(self.model.encode_text(tokens), dim=-1)
        return (img_e @ txt_e.T).squeeze()

    def forward(self, img_L, img_R, cos_lr):
        if not self.clip_available:
            return cos_lr.new_tensor(0.0)
        t_L = self.get_clip_sim(img_L, self.prompt_L)
        t_R = self.get_clip_sim(img_R, self.prompt_R)
        s_star = 1 - torch.abs(t_L - t_R)
        return F.mse_loss(cos_lr, s_star)


# ---------------- DenseNet121 Siamese ----------------
class SiameseDenseLBP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.encoder = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        enc_dim = 1024

        # Projection + dropout (matches training checkpoint)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Classifier (matches training checkpoint exactly)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

        # Adjust for 4-channel input (RGB + LBP)
        old_conv = self.encoder[0]
        self.encoder[0] = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            self.encoder[0].weight[:, :3] = old_conv.weight
            self.encoder[0].weight[:, 3:] = old_conv.weight[:, 0:1]

    def encode(self, x):
        f = self.encoder(x)
        f = self.pool(f)
        f = self.proj(f)
        return F.normalize(f, dim=-1)

    def forward(self, x):
        x = compute_lbp_batch(x)
        xl, xr = split_left_right(x)
        fl, fr = self.encode(xl), self.encode(xr)
        cos = F.cosine_similarity(fl, fr)
        z = torch.cat([fl, fr, torch.abs(fl - fr)], dim=-1)
        return self.classifier(z), cos, xl[:, :3], xr[:, :3]
