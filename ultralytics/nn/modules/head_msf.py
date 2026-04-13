# Ultralytics YOLO 🚀, AGPL-3.0 license
"""多尺度融合优化的JDE头部模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import JDE
from .conv import Conv


class JDE_MSF(JDE):
    """
    多尺度特征融合的JDE头部
    
    改进点：
    1. 跨尺度ReID特征融合：让不同尺度的ReID特征相互增强
    2. 自适应特征聚合：根据特征重要性动态融合
    3. 轻量级设计：最小化额外计算开销
    
    适用场景：
    - 目标尺度变化大（远近距离）
    - 遮挡严重的场景
    - 需要提升HOTA指标
    """

    def __init__(self, nc=80, embed_dim=128, ch=(), fusion_type='adaptive'):
        """
        初始化多尺度融合JDE头部
        
        Args:
            nc: 类别数
            embed_dim: ReID特征维度
            ch: 各尺度输入通道数 (P3, P4, P5)
            fusion_type: 融合方式 ['adaptive', 'concat', 'add']
        """
        super().__init__(nc, embed_dim, ch)
        self.fusion_type = fusion_type
        
        # 多尺度ReID特征融合模块
        if fusion_type == 'adaptive':
            # 自适应权重融合（推荐）
            # self.scale_weights = nn.Parameter(torch.ones(self.nl) / self.nl)
            # 使用 register_buffer 确保自动同步到正确设备
            self.scale_weights = nn.Parameter(torch.ones(self.nl, dtype=torch.float32) / self.nl)
            # 特征对齐卷积（统一不同尺度特征的语义）
            c4 = max(ch[0] // 4, self.embed_dim)
            self.feat_align = nn.ModuleList([
                nn.Sequential(
                    Conv(self.embed_dim, c4, 1),
                    Conv(c4, self.embed_dim, 1)
                ) for _ in range(self.nl)
            ])
            
        elif fusion_type == 'concat':
            # 拼接融合（特征维度会增加）
            self.fusion_conv = nn.Sequential(
                Conv(self.embed_dim * self.nl, self.embed_dim * 2, 1),
                Conv(self.embed_dim * 2, self.embed_dim, 1)
            )
            
        elif fusion_type == 'add':
            # 简单相加（最轻量）
            pass

    def forward(self, x):
        """
        前向传播，加入多尺度ReID特征融合
        
        Args:
            x: 来自neck的多尺度特征 [P3, P4, P5]
            
        Returns:
            融合后的检测+ReID输出
        """
        # 先提取各尺度的ReID特征（不concat）
        reid_feats = []
        det_cls_feats = []
        
        for i in range(self.nl):
            # 检测框回归
            box_feat = self.cv2[i](x[i])
            # 分类
            cls_feat = self.cv3[i](x[i])
            # ReID特征
            reid_feat = self.cv4[i](x[i])
            
            reid_feats.append(reid_feat)
            det_cls_feats.append((box_feat, cls_feat))
        
        # 多尺度ReID特征融合
        if self.fusion_type == 'adaptive':
            # 自适应融合：为每个尺度学习权重
            fused_reid_feats = []
            device = reid_feats[0].device
            scale_weights = self.scale_weights.to(device)
            weights = F.softmax(scale_weights, dim=0)
            
            for i in range(self.nl):
                # 将其他尺度特征resize到当前尺度
                h, w = reid_feats[i].shape[2:]
                multi_scale_feats = []
                
                for j in range(self.nl):
                    if i == j:
                        multi_scale_feats.append(reid_feats[j] * weights[j])
                    else:
                        # 上采样或下采样到目标尺度
                        resized = F.interpolate(
                            reid_feats[j], 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        # 特征对齐
                        aligned = self.feat_align[j](resized)
                        multi_scale_feats.append(aligned * weights[j])
                
                # 融合
                fused = sum(multi_scale_feats)
                fused_reid_feats.append(fused)
                
        elif self.fusion_type == 'concat':
            # 拼接融合
            fused_reid_feats = []
            for i in range(self.nl):
                h, w = reid_feats[i].shape[2:]
                # 将所有尺度resize到当前尺度并拼接
                resized_feats = [
                    F.interpolate(reid_feats[j], size=(h, w), mode='bilinear', align_corners=False)
                    for j in range(self.nl)
                ]
                concat_feat = torch.cat(resized_feats, dim=1)
                fused = self.fusion_conv(concat_feat)
                fused_reid_feats.append(fused)
                
        elif self.fusion_type == 'add':
            # 简单相加融合
            fused_reid_feats = []
            for i in range(self.nl):
                h, w = reid_feats[i].shape[2:]
                # 将所有尺度resize到当前尺度并相加
                resized_feats = [
                    F.interpolate(reid_feats[j], size=(h, w), mode='bilinear', align_corners=False)
                    for j in range(self.nl)
                ]
                fused = sum(resized_feats) / self.nl
                fused_reid_feats.append(fused)
        
        # 组合最终输出
        outputs = []
        for i in range(self.nl):
            box_feat, cls_feat = det_cls_feats[i]
            outputs.append(torch.cat((box_feat, cls_feat, fused_reid_feats[i]), 1))
        
        if self.training:
            return outputs
        
        y = self._inference(outputs)
        return y if self.export else (y, outputs)


class JDE_ASPP(JDE):
    """
    带ASPP（空洞空间金字塔池化）的JDE头部
    
    改进点：
    1. 在ReID分支加入ASPP模块，增强多尺度感受野
    2. 捕获不同尺度的上下文信息
    3. 对遮挡和尺度变化更鲁棒
    """

    def __init__(self, nc=80, embed_dim=128, ch=(), dilation_rates=[6, 12, 18]):
        """
        初始化带ASPP的JDE头部
        
        Args:
            nc: 类别数
            embed_dim: ReID特征维度
            ch: 各尺度输入通道数
            dilation_rates: 空洞卷积的膨胀率
        """
        super().__init__(nc, embed_dim, ch)
        
        # 用ASPP替换原始的cv4
        c4 = max(ch[0] // 4, self.embed_dim)
        self.cv4 = nn.ModuleList([
            ASPP(x, self.embed_dim, dilation_rates) for x in ch
        ])


class ASPP(nn.Module):
    """空洞空间金字塔池化模块"""
    
    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1卷积分支
        self.conv1x1 = Conv(in_channels, out_channels // 4, 1)
        
        # 多个空洞卷积分支
        self.aspp_branches = nn.ModuleList([
            Conv(in_channels, out_channels // 4, 3, p=rate, d=rate)
            for rate in dilation_rates
        ])
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1),  # 直接使用Conv2d
            nn.SiLU(inplace=True)  # 添加激活函数
        )
        
        # 融合卷积
        total_channels = out_channels // 4 * (len(dilation_rates) + 2)
        self.fusion = nn.Sequential(
            Conv(total_channels, out_channels, 1),
            Conv(out_channels, out_channels, 3),
            nn.Conv2d(out_channels, out_channels, 1)
        )
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # 1x1卷积
        feat1 = self.conv1x1(x)
        
        # 空洞卷积
        aspp_feats = [branch(x) for branch in self.aspp_branches]
        
        # 全局池化
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # 拼接所有分支
        concat_feat = torch.cat([feat1] + aspp_feats + [global_feat], dim=1)
        
        # 融合
        out = self.fusion(concat_feat)
        return out


class JDE_ChannelAttention(JDE):
    """
    带通道注意力的JDE头部
    
    改进点：
    1. 在ReID分支加入通道注意力机制
    2. 自动学习哪些通道对ReID更重要
    3. 轻量级，几乎不增加计算量
    """

    def __init__(self, nc=80, embed_dim=128, ch=(), reduction=16):
        super().__init__(nc, embed_dim, ch)
        
        # 为每个尺度的ReID分支添加通道注意力
        c4 = max(ch[0] // 4, self.embed_dim)
        self.cv4 = nn.ModuleList([
            nn.Sequential(
                Conv(x, c4, 3),
                ChannelAttentionModule(c4, reduction),
                Conv(c4, c4, 3),
                ChannelAttentionModule(c4, reduction),
                nn.Conv2d(c4, self.embed_dim, 1)
            ) for x in ch
        ])


class ChannelAttentionModule(nn.Module):
    """通道注意力模块（轻量级）"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention
