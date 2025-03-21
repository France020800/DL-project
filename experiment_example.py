from main import main
import numpy as np

'''
Experiment with:
 - Dataset = CIFAR10
 - Model = ResNet152
 - Epochs = 50
 - EWC = True
'''

losses_over_classes = []
auc_scores_over_classes = []

for i in range(0, 10):
    print(f'Start training on {i} class')
    losses, auc_scores = main('resnet152', dataset='cifar10', saved_name=None, epochs=50, normal_class=i, batch_size=32, ewc=True, OE=False, verbose=True)
    losses_over_classes.append(losses[len(losses) - 1])
    auc_scores_over_classes.append(auc_scores[len(auc_scores) - 1])
    print(f'Class: {i}\nScore: {auc_scores[len(auc_scores) - 1]}\n')

loss_mean = np.mean(losses_over_classes)
auc_mean = np.mean(auc_scores_over_classes)
print(f'Average Loss: {loss_mean}\nAverage auc: {auc_mean}\n')