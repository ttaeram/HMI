# clip_wrapper.py
import torch
import torch.nn as nn
from ml_mobileclip.mobileclip.clip import CLIP

class MobileCLIPWrapper(nn.Module):
    def __init__(self, config_dict, device):
        super().__init__()
        self.clip = CLIP(config_dict).to(device).eval()

    def forward(self, image_tensor, text_tensor=None):
        image_features = self.clip.image_encoder(image_tensor)
        text_features = self.clip.text_encoder(text_tensor) if text_tensor is not None else None
        return image_features, text_features
