import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import models, transforms
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
from transformers import AutoModel
from models_vit_2 import RETFound_mae

# Logging 설정
dir_path = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_path, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now():%Y%m%d_%H%M%S}.log")

# CLAHE 전처리 클래스
class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = self.clahe.apply(yuv[:,:,0])
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(rgb)
    

# ADAM Dataset (CLAHE 저장 기능 포함)
class ADAMDataset(Dataset):
    def __init__(self, csv_file: str, fundus_dir: str, macula_dir: str, transform=None, save_clahe_n: int = 0):
        self.data = pd.read_csv(csv_file)
        self.fundus_dir = fundus_dir
        self.macula_dir = macula_dir
        self.transform = transform
        self.save_clahe_n = save_clahe_n  # 처음 N개 항목에 대해 CLAHE 이미지 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data.iloc[idx]['filename']
        label = self.data.iloc[idx]['label']
        
        # fundus 이미지 로드
        fundus = Image.open(os.path.join(self.fundus_dir, fname)).convert('RGB')
        
        # macula 이미지 로드
        macula = Image.open(os.path.join(self.macula_dir, fname)).convert('RGB')

        if self.transform:
            # transform이 Compose이면, 첫 단계 CLAHE와 나머지 분리
            if isinstance(self.transform, transforms.Compose) and isinstance(self.transform.transforms[0], CLAHE):
                clahe_tf = self.transform.transforms[0]
                rest_tf = transforms.Compose(self.transform.transforms[1:])
                
                # fundus 이미지 처리
                fundus_clahe = clahe_tf(fundus)
                if idx < self.save_clahe_n:
                    save_dir = os.path.join(dir_path, 'clahe_samples')
                    os.makedirs(save_dir, exist_ok=True)
                    fundus_clahe.save(os.path.join(save_dir, f"clahe_{idx}_{fname}"))
                fundus = rest_tf(fundus_clahe)
                
                # macula 이미지 처리
                macula_clahe = clahe_tf(macula)
                macula = rest_tf(macula_clahe)
            else:
                fundus = self.transform(fundus)
                macula = self.transform(macula)

        return fundus, macula, torch.tensor(label, dtype=torch.float32), idx
    

# Top-k gating 함수
def top_k_gating(logits: torch.Tensor, k: int = 1, dim: int = -1) -> torch.Tensor:
    """
    Top-k gating 함수.
    Args:
        logits (Tensor): 입력 logits, shape: [..., num_experts]
        k (int): 사용할 expert의 개수.
        dim (int): gating을 적용할 차원.
    Returns:
        Tensor: gating 결과, 동일한 shape.
    """
    topk_logits, topk_indices = torch.topk(logits, k, dim=dim)
    gating_weights = torch.zeros_like(logits)
    # topk logits 에 소프트맥스 적용하여 가중치 할당
    gating_weights.scatter_(dim, topk_indices, F.softmax(topk_logits, dim=dim))
    return gating_weights


# AMD Network with MoE Fusion
class AMD_NET(nn.Module):
    def __init__(self, num_experts=4, hidden_dim=128, k=1):
        super().__init__()
        self.fe = SwinTFeatureExtractor(pretrained=True)  # 하나의 FE 모듈
        feat_dim = 768

        # 2개 피처(f1,f2)를 다시 feat_dim 으로 줄이는 fusion 레이어
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

        # hard-gating MoE: fused vector h 를 입력으로 받음
        self.moe = HardGatingMoE(input_dim=feat_dim,
                                 hidden_dim=hidden_dim,
                                 num_experts=num_experts,
                                 k=k)

        # MoE 후 분류기
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x_fundus: torch.Tensor,
                      x_macula: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) 두 브랜치로부터 각각 피처 추출
        f1 = self.fe(x_fundus)   # [B, feat_dim]
        f2 = self.fe(x_macula)   # [B, feat_dim]

        # 2) 선형 fusion
        h = torch.cat([f1, f2], dim=1)      # [B, feat_dim*2]
        h = self.fusion(h)                  # [B, feat_dim]

        # 3) Hard-gating MoE
        fused, mask = self.moe(h)           # fused: [B, feat_dim], mask: [B, num_experts]

        # 4) 분류기
        logit = self.classifier(fused)      # [B,1]
        return logit.squeeze(1), mask
    



# SwinT Feature Extractor
class SwinTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        swin = models.swin_t(pretrained=pretrained)
        modules = list(swin.children())[:-1]
        self.backbone = nn.Sequential(*modules)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # [B,C,1,1]
        return feat.view(feat.size(0), -1)  # [B,C]

# RETFound Feature Extractor
class RETFoundFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, model_name="RetFound/retfound-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(model_name)
        # RETFound는 마지막 output이 tuple (last_hidden_state, pooled_output, ...) 구조입니다.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224] (이미지)
        outputs = self.backbone(x)  # outputs: BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            feat = outputs.pooler_output    # [B, feature_dim]
        else:
            feat = outputs.last_hidden_state[:, 0, :]  # [B, feature_dim] (CLS 토큰)
        return feat

# class RETFoundFeatureExtractor(nn.Module):
#     def __init__(self, weights_path=r"C:\Users\nayeon\Desktop\research\medical\amd\code\multi-scale\RETFound\RETFound_cfp_weights.pth", device='cpu'):
#         super().__init__()
#         self.backbone = RETFound_mae()
#         checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
#         self.backbone.load_state_dict(checkpoint['model'], strict=False)

#     def forward(self, x):
#         # forward_features가 [B,1024] 반환 (global_pool=False 시 [CLS] 토큰)
#         return self.backbone.forward_features(x)
class RETFoundFeatureExtractor(nn.Module):
    def __init__(self, weights_path=r"/home/gpuadmin/nayeon/code/multi-scale/RETFound/RETFound_cfp_weights.pth", device='cpu', embed_dim=1024):
        super().__init__()
        # embed_dim을 파라미터로 받아서 초기화
        self.backbone = RETFound_mae(embed_dim=embed_dim)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        self.backbone.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x):
        # forward_features 출력은 [B, N_patches+1, embed_dim] 형태 (CLS 토큰 포함)
        x = self.backbone.forward_features(x)  # shape: [B, seq_len, 768]

        # 일반적으로 [CLS] 토큰만 사용 (첫 번째 토큰)
        cls_token = x[:, 0, :]  # shape: [B, 768]

        return cls_token



# Hard Gating MoE
class HardGatingMoE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)                        # [B, num_experts]
        gating_weights = top_k_gating(logits, k=self.k, dim=-1)  # [B, num_experts]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, num_experts, input_dim]
        gating_weights = gating_weights.unsqueeze(-1)  # [B, num_experts, 1]
        fused = (expert_outs * gating_weights).sum(dim=1)  # [B, input_dim]
        return fused, gating_weights.squeeze(-1)



