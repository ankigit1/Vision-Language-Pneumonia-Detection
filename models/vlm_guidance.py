# models/vlm_guidance.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageGuidance(nn.Module):
    """
    Lightweight wrapper for CLIP-based laterality guidance.
    Computes s* = 1 - |t_L - t_R|
    """

    def __init__(self, device):
        super().__init__()
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self.model.eval()
            self.prompts = {
                "left": clip.tokenize(["left lung opacity"]).to(device),
                "right": clip.tokenize(["right lung opacity"]).to(device),
            }
            self.ok = True
        except Exception as e:
            print("⚠️ CLIP unavailable:", e)
            self.ok = False

    @torch.no_grad()
    def forward(self, img_left, img_right, cos_sim):
        if not self.ok:
            return cos_sim.new_tensor(0.0)

        import clip
        img_left = F.interpolate(img_left[:, :3], size=224)
        img_right = F.interpolate(img_right[:, :3], size=224)

        v_left = F.normalize(self.model.encode_image(img_left), dim=-1)
        v_right = F.normalize(self.model.encode_image(img_right), dim=-1)
        t_left = F.normalize(self.model.encode_text(self.prompts["left"]), dim=-1)
        t_right = F.normalize(self.model.encode_text(self.prompts["right"]), dim=-1)

        sL = (v_left @ t_left.T).squeeze()
        sR = (v_right @ t_right.T).squeeze()
        s_star = 1 - torch.abs(sL - sR)
        return F.mse_loss(cos_sim, s_star)
