import torch
import torch.nn as nn
import timm
from torchvision.transforms.functional import resize
import torch.nn.functional as F
class DimReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimReduction, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reduce(x)
    
class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.v1 = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.init_param()

    def forward(self, x1, x2, x3):
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y = torch.cat((y1,y2,y3), dim=1) + y4
        y = self.res(y)
        return y

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class CustomTwinsSVTLarge(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomTwinsSVTLarge, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
        self.dim_red1 = DimReduction(320, 128)
        self.dim_red2 = DimReduction(128, 64)
        self.dim_red3 = DimReduction(64, 16)
        self.dim_red4 = DimReduction(16, 3)  # Final reduction to 3 channels
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # 假设原图大小是EfficientNet输出的32倍
        self.dim_mix = DimReduction(6, 3)
        self.twins = timm.create_model('twins_svt_large', pretrained=pretrained, num_classes=0)
        self.regression = Regression()  # 初始化 Regression 模块
    def forward(self, x):
        features = self.efficientnet(x)
        efficientnet_output = self.dim_red1(features[-1])

        # 计算上采样倍率，以将EfficientNet的输出还原到原图大小
        original_size = x.size()[2:]  # 原始图像尺寸 (H, W)
        scale_factor = (original_size[0] / efficientnet_output.size(2), original_size[1] / efficientnet_output.size(3))

        # 上采样
        upsampled_efficientnet_output = F.interpolate(efficientnet_output, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        #逐步降維
        upsampled_efficientnet_output = self.dim_red2(upsampled_efficientnet_output)
        upsampled_efficientnet_output = self.dim_red3(upsampled_efficientnet_output)
        upsampled_efficientnet_output = self.dim_red4(upsampled_efficientnet_output)
        
        # 融合原图和上采样后的EfficientNet输出
        fused = torch.cat([x, upsampled_efficientnet_output], dim=1)
        

        x_3_channels=self.dim_mix(fused)
        # 使用卷积层将融合后的6通道图像转换为3通道
        

        # 适配Twins模型的输入
        x_t = x_3_channels
        stage_outputs = []

        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.twins.patch_embeds, self.twins.pos_drops, self.twins.blocks, self.twins.pos_block)):
            x_t, size = embed(x_t)  # 假设size是一个包含(H, W)的元组
            x_t = drop(x_t)
            for j, blk in enumerate(blocks):
                x_t = blk(x_t, size)
                if j == 0:
                    x_t = pos_blk(x_t, size)
            
            # 动态调整形状，适应当前阶段的特征图大小
            B, _, C = x_t.shape  # 获取当前批次大小和通道数
            H, W = size  # 从embed返回的size中获取特征图的高度和宽度
            x_t = x_t.permute(0, 2, 1).reshape(B, C, H, W)  # 调整形状为[B, C, H, W]
            
            stage_outputs.append(x_t)

        # 应用 Regression 到最后三个阶段的输出
        mu = self.regression(stage_outputs[-3], stage_outputs[-2], stage_outputs[-1])  # 根据需要传递正确的参数
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)

        return mu, mu_normed
