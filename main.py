import torch
import torch.optim as optim
from copy import deepcopy
from my_resnet import MyResNet
from losses import EWCLoss, CompactnessLoss
import utils


def train_model(model, train_loader, test_loader, device, ewc, ewc_loss, epochs, verbose):
    model.eval()
    is_learning = True
    if verbose:
        auc, feature_space = utils.get_score_verbose(model, device, train_loader, test_loader)
    else:
        auc, feature_space = utils.get_score(model, device, train_loader, test_loader)
    print(f'Epoch: 0, AUROC: {auc}')
    
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    losses = []
    auc_scores = []

    # epoch = 0
    # best_auc = 0.0
    # while is_learning and epoch < epochs:
    for epoch in epochs:
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss)
        if verbose:
            print(f'Epoch: {epoch + 1}, Loss: {running_loss}')
            auc, feature_space = utils.get_score_verbose(model, device, train_loader, test_loader)
            print(f'Epoch: {epoch + 1}, AUROC: {auc}')
        else:
            auc, feature_space = utils.get_score(model, device, train_loader, test_loader)
            print(f'AUC: {auc} on {epoch + 1}')
        losses.append(running_loss)
        auc_scores.append(auc)

        ''''
        if best_auc < auc:
            best_auc = auc
        else:
            is_learning = False
        epoch += 1
        '''
    
    del train_loader
    del test_loader
    return losses, auc_scores


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()
              
        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def train_model_OE(model, train_loader, outliers_loader, test_loader, device, epochs, verbose):
    is_learning = True
    model.change_last_layer(1)
    model = model.to(device)
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    bce = torch.nn.BCELoss()

    losses = []
    auc_scores = []
    # epoch = 0
    # best_auc = 0.0
    # while is_learning and epoch < epochs:
    for epoch in epochs:
        running_loss = run_epoch_OE(model, train_loader, outliers_loader, optimizer, bce, device)
        if verbose:
            print(f'Epoch: {epoch + 1}, Loss: {running_loss}')
            auc = utils.get_score_OE(model, device, test_loader)
            print(f'Epoch: {epoch + 1}, AUROC: {auc}')
        else:
            auc = utils.get_score_OE_verbose(model, device, test_loader)
            print(f'AUC: {auc} on {epoch + 1}')

        ''''
        if best_auc < auc:
            best_auc = auc
        else:
            is_learning = False
        epoch += 1
        '''

        losses.append(running_loss)
        auc_scores.append(auc)

    del train_loader
    del test_loader
    return losses, auc_scores


def run_epoch_OE(model, train_loader, outliers_loader, optimizer, bce, device):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        imgs = imgs.to(device)

        out_imgs, _ = next(iter(outliers_loader))

        outlier_im = out_imgs.to(device)

        optimizer.zero_grad()

        pred, _ = model(imgs)
        outlier_pred, _ = model(outlier_im)

        batch_1 = pred.size()[0]
        batch_2 = outlier_pred.size()[0]

        labels = torch.zeros(size=(batch_1 + batch_2,), device=device)
        labels[batch_1:] = torch.ones(size=(batch_2,))

        loss = bce(torch.sigmoid(torch.cat([pred, outlier_pred])), labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def main(resnet_type, dataset, saved_name=None, epochs=15, normal_class=0, batch_size=32, ewc=True, OE=False, verbose=True):
    torch.cuda.empty_cache()
    if verbose:
        print(f'Dataset: {dataset}, Normal Class: {normal_class}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyResNet(model_name=resnet_type)
    model.freeze_parameters()

    train_loader, test_loader = utils.get_loaders(dataset=dataset, normal_class=normal_class, batch_size=batch_size)
    
    ewc_loss = None
    if ewc and not OE:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        frozen_model.freeze_model()
        try:
            fisher = torch.load('./data/fisher_diagonal.pth')
        except:
            fisher = utils.compute_fisher_information(train_loader, device)
            torch.save(fisher, './data/fisher_diagonal.pth')
        ewc_loss = EWCLoss(frozen_model, fisher)
    
    if OE:
        outliers_loader = utils.get_outliers_loader(batch_size)
        losses, auc_scores = train_model_OE(model, train_loader, outliers_loader, test_loader, device, epochs)
    else:
        model = model.to(device)
        losses, auc_scores = train_model(model, train_loader, test_loader, device, ewc, ewc_loss, epochs, verbose)
    
    if saved_name:
        torch.save(model, f'models/{saved_name}')

    torch.cuda.empty_cache()
    del model
    return losses, auc_scores

