from my_resnet import MyResNet
import utils as utils
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyResNet(model_name='resnet152')
model.freeze_parameters()
model = model.to(device)
model.eval()

auc_scores = []

for i in range(10):
    train_loader, test_loader = utils.get_loaders(dataset='cifar10', normal_class=i, batch_size=32)

    auc, _ = utils.get_score(model, device, train_loader, test_loader)
    auc_scores.append(auc)
    print(f'Class: {i}\nScore: {auc}')

auc_mean = np.mean(auc_scores)
print(f'\nMean: {auc_mean}')