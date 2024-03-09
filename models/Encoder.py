import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in feature_extractor.parameters():
            param.requires_grad_(False)
        
        modules = list(feature_extractor.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.feature_extractor(images)                         #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features