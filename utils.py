import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import faiss
from sklearn.metrics import roc_auc_score
from my_resnet import MyResNet


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def plot(losses, auc_scores, info):
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', marker='o', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.xticks(range(1, len(losses) + 1))
    plt.legend()
    
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(auc_scores) + 1), auc_scores, label='AUC Score', marker='o', color='green')
    plt.title('AUC Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.xticks(range(1, len(losses) + 1))
    plt.legend()
    
    plt.grid(True)
    plt.show()
    
    print(info)
    print(f'loss: {losses[-1]}')
    print(f'AUC: {auc_scores[-1]}')


def knn_score(train_set, test_set, n_neighbours=2):
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_score_verbose(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        try:
            test_labels = test_loader.dataset.targets
        except:
            test_labels = [test_loader.dataset.dataset.targets[i] for i in test_loader.dataset.indices]

    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in train_loader:
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in test_loader:
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        try:
            test_labels = test_loader.dataset.targets
        except:
            test_labels = [test_loader.dataset.dataset.targets[i] for i in test_loader.dataset.indices]

    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def get_score_OE_verbose(model, device, test_loader):
    model.eval()
    anom_labels = []
    predictions = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Test prediction'):
            imgs, labels = imgs.to(device), labels.numpy()
            pred, _ = model(imgs)
            pred = torch.sigmoid(pred)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                predictions.append(pred[j].detach().cpu().numpy())
                anom_labels.append(labels[j])

    test_set_predictions = np.array(predictions)
    test_labels = np.array(anom_labels)

    auc = roc_auc_score(test_labels, test_set_predictions)

    return auc


def get_score_OE(model, device, test_loader):
    model.eval()
    anom_labels = []
    predictions = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.numpy()
            pred, _ = model(imgs)
            pred = torch.sigmoid(pred)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                predictions.append(pred[j].detach().cpu().numpy())
                anom_labels.append(labels[j])

    test_set_predictions = np.array(predictions)
    test_labels = np.array(anom_labels)

    auc = roc_auc_score(test_labels, test_set_predictions)

    return auc


def get_loaders(dataset, normal_class, batch_size):
    if dataset in ['cifar10', 'cifar100', 'fashion', 'mnist']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "cifar100":
            ds = torchvision.datasets.CIFAR100
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "mnist":
            ds = torchvision.datasets.MNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == normal_class
        testset.targets = [int(t != normal_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
        
    elif dataset == 'cats_vs_dogs':
        images_dir = 'data/catVSdog/PetImages'
        dataset = torchvision.datasets.ImageFolder(root=images_dir, transform=transform_color, is_valid_file=is_valid_image)
        
        normal_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
        anomalous_indices = [i for i, (_, label) in enumerate(dataset) if label != 0]
        
        random.seed(42)
        train_indices = random.sample(normal_indices, 10000)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        
        remaining_normal_indices = list(set(normal_indices) - set(train_indices))
        
        test_normal_indices = random.sample(remaining_normal_indices, 1000)
        test_anomalous_indices = random.sample(anomalous_indices, 4000)
        test_indices = test_normal_indices + test_anomalous_indices
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

        return train_loader, test_loader
        
    else:
        print('Unsupported Dataset')
        exit()


def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader


def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406])  # Mean used in normalization
    std = torch.tensor([0.229, 0.224, 0.225])  # Std used in normalization
    img = img * std[:, None, None] + mean[:, None, None]  # Denormalize
    return img.clamp(0, 1)  # Ensure pixel values are in [0, 1]


def display_image(dataset):
    # Get a batch from train_loader
    data_iter = iter(dataset)
    images, labels = next(data_iter)
    
    # Display the first few images
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Adjust the number of images to display
    for i, ax in enumerate(axes):
        img = denormalize(images[i])  # Denormalize the image
        img = img.permute(1, 2, 0)  # Change tensor shape from CxHxW to HxWxC for plotting
        ax.imshow(img.numpy())
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")  # Hide axis
    
    plt.tight_layout()
    plt.show()


def compute_fisher_information(dataloader, device):
    """
    Compute the diagonal Fisher Information Matrix for a model.
    """
    model = MyResNet(model_name='resnet152')
    for param in model.parameters():
            param.requires_grad = True
        
    model.eval()
    model = model.to(device)
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    for inputs, labels in tqdm(dataloader, desc="Computing Fisher"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = model(inputs)
        log_probs = torch.log_softmax(outputs, dim=1)  # Log-likelihood

        loss = torch.nn.functional.nll_loss(log_probs, labels)

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

    num_samples = len(dataloader.dataset)
    for name in fisher:
        fisher[name] /= num_samples
        
    return fisher