import torch
import torch.nn as nn
from torchvision import models

class MultimodalFusionNet(nn.Module):
    def __init__(self, tabular_dim):
        super(MultimodalFusionNet, self).__init__()
        
        # 1. Vision Branch: EfficientNetV2
        # It converts an image into a vector of 1280 numbers
        self.vision_model = models.efficientnet_v2_s(weights='DEFAULT')
        self.vision_model.classifier = nn.Identity() 

        # 2. Tabular Branch: Processes patient labs/vitals
        # It converts clinical data into a vector of 32 numbers
        self.clinical_model = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 3. Fusion Layer: Glues 1280 + 32 = 1312 features together
        self.fusion_layer = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() # Output is a probability between 0 and 1
        )

    def forward(self, img, tab):
        v_feat = self.vision_model(img)
        c_feat = self.clinical_model(tab)
        # Concatenate features side-by-side
        combined = torch.cat((v_feat, c_feat), dim=1)
        return self.fusion_layer(combined)