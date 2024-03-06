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
class CustomEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomEfficientNet, self).__init__()
        # 加載 EfficientNet 模型
        self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
        # 記錄每個階段的輸出通道數
        self.stage_channels = [self.base_model.feature_info.channels()[i] for i in range(len(self.base_model.stages) - 3, len(self.base_model.stages))]
        # 降低每個階段輸出的維度
        self.dim_reduction_layers = nn.ModuleList([
            DimReduction(in_channels, 3) for in_channels in self.stage_channels
        ])
        # 用於融合輸出和原始圖像的卷積層
        self.merge_conv = nn.Conv2d(sum(self.stage_channels) + 3, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
    # 原始圖像大小
        origin = x
        orig_size = x.size()[2:]
        # 從最後幾個階段獲取特徵
        all_features = self.base_model(x)
        # 根據 stage_channels 選擇最後幾個階段的特徵
        stage_features = all_features[-(len(self.stage_channels)):]
        # 上採樣並降低每個階段輸出的維度
        upsampled_features = []
        for feature, dim_reduce in zip(stage_features, self.dim_reduction_layers):
            upsampled_feature = F.interpolate(feature, size=orig_size, mode='bilinear', align_corners=False)
            reduced_feature = dim_reduce(upsampled_feature)
            upsampled_features.append(reduced_feature)
        # 將上採樣的特徵與原始圖像連接起來
        merged_features = torch.cat([origin] + upsampled_features, dim=1)
        # 使用卷積層進行融合
        merged_output = self.merge_conv(merged_features)
        return merged_output


class CustomTwinsSVTLarge(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomTwinsSVTLarge, self).__init__()
        self.customEfficient=CustomEfficientNet()
        self.twins = timm.create_model('twins_svt_large', pretrained=pretrained, num_classes=0)
        self.regression = Regression()  # 初始化 Regression 模块
    def forward(self, x):
        x = self.customEfficient(x)
        stage_outputs = []

        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.twins.patch_embeds, self.twins.pos_drops, self.twins.blocks, self.twins.pos_block)):
            x, size = embed(x)  # 假设size是一个包含(H, W)的元组
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            
            # 动态调整形状，适应当前阶段的特征图大小
            B, _, C = x.shape  # 获取当前批次大小和通道数
            H, W = size  # 从embed返回的size中获取特征图的高度和宽度
            x = x.permute(0, 2, 1).reshape(B, C, H, W)  # 调整形状为[B, C, H, W]
            
            stage_outputs.append(x)

        # 应用 Regression 到最后三个阶段的输出
        mu = self.regression(stage_outputs[-3], stage_outputs[-2], stage_outputs[-1])  # 根据需要传递正确的参数
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)

        return mu, mu_normed