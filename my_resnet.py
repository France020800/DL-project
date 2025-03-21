import torch
import torch.nn as nn

class MyResNet(nn.Module):

    def __init__(self, model_name='resnet152'):
        super(MyResNet, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc 

    def freeze_parameters(self):
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False

    def change_last_layer(self, num_out_classes):
        num_features = self.fc.in_features
        self.fc = nn.Linear(num_features, num_out_classes)

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        
        f = self.avgpool(f)
        out_features = torch.flatten(f, 1)
        out = self.fc(out_features)
        return out.squeeze(), out_features.squeeze()