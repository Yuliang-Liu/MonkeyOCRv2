"""
Take Performer as T2T Transformer
"""
import math
import torch
import torch.nn as nn
import numpy as np


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1, gamma=False, init_values=1e-4):
        super().__init__()
        self.head_dim = in_dim // head_cnt
        self.emb = in_dim
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division
        self.drop_path = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.head_dim * kernel_ratio)
        self.w = torch.randn(head_cnt, self.m, self.head_dim)
        for i in range(self.head_cnt):
            self.w[i] = nn.Parameter(nn.init.orthogonal_(self.w[i]) * math.sqrt(self.m), requires_grad=False)
        self.w.requires_grad_(False)

        if gamma:
            self.gamma1 = nn.Parameter(init_values * torch.ones((self.emb)),requires_grad=True)
        else:
            self.gamma1 = 1

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, H, N, hs)
        # w = (H, m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2
        wtx = torch.einsum('bhti,hmi->bhtm', x.float(), self.w.to(x.device))

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        B, N, C = x.shape
        kqv = self.kqv(x).reshape(B, N, 3, self.head_cnt, self.head_dim).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2] # (B, H, T, hs)

        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, H, T, m), (B, H, T, m)
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=-1)  # (B, H, T, m) * (B, H, m) -> (B, H, T, 1)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)  # (B, H, emb, m)
        y = torch.einsum('bhti,bhni->bhtn', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)  # (B, H, T, emb)/Diag

        # skip connection

        y = y.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.emb)

        y = v + self.dp(self.gamma1 * self.proj(y))  # same as token_transformer, use v as skip connection

        return y

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x