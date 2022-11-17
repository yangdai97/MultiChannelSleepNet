import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model import Transformer
from early_stop_tool import EarlyStopping
from data_loader import data_generator
from args import Config, Path


def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)        # CPU
    torch.cuda.manual_seed(seed)   # GPU


def test(model, test_loader, config):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    pred = []
    label = []

    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(config.device)
            target = target.to(config.device)
            data, target = Variable(data), Variable(target)

            output = model(data)
            test_loss += criterion(output, target.long()).item()

            pred.extend(np.argmax(output.data.cpu().numpy(), axis=1))
            label.extend(target.data.cpu().numpy())

        accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)

    return accuracy, test_loss


def train(save_all_checkpoint=False):
    config = Config()
    path = Path()

    dataset, labels, val_loader = data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)

    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)
        if not os.path.exists('./Kfold_models/fold{}'.format(fold)):
            os.makedirs('./Kfold_models/fold{}'.format(fold))

        X_train, X_test = dataset[train_idx], dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False)

        model = Transformer(config)
        model = model.to(config.device)

        criterion = nn.CrossEntropyLoss()

        # AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        # apply early_stop. If you want to view the full training process, set the save_all_checkpoint True
        early_stopping = EarlyStopping(patience=20, verbose=True, save_all_checkpoint=save_all_checkpoint)

        # evaluating indicator
        train_ACC = []
        train_LOSS = []
        test_ACC = []
        test_LOSS = []
        val_ACC = []
        val_LOSS = []

        for epoch in range(config.num_epochs):
            running_loss = 0.0
            correct = 0

            model.train()

            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, target) in loop:
                data = data.to(config.device)
                target = target.to(config.device)
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target.long())

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                train_acc_batch = np.sum(np.argmax(np.array(output.data.cpu()), axis=1) == np.array(target.data.cpu())) / (target.shape[0])
                loop.set_postfix(train_acc=train_acc_batch, loss=loss.item())
                correct += np.sum(np.argmax(np.array(output.data.cpu()), axis=1) == np.array(target.data.cpu()))

            train_acc = correct / len(train_loader.dataset)
            test_acc, test_loss = test(model, test_loader, config)
            val_acc, val_loss = test(model, val_loader, config)
            print('Epoch: ', epoch,
                  '| train loss: %.4f' % running_loss, '| train acc: %.4f' % train_acc,
                  '| val acc: %.4f' % val_acc, '| val loss: %.4f' % val_loss,
                  '| test acc: %.4f' % test_acc, '| test loss: %.4f' % test_loss)

            train_ACC.append(train_acc)
            train_LOSS.append(running_loss)
            test_ACC.append(test_acc)
            test_LOSS.append(test_loss)
            val_ACC.append(val_acc)
            val_LOSS.append(val_loss)

            # Check whether to continue training. If save_all_checkpoint=False, the model name will be ‘model.pkl'
            early_stopping(val_acc, model, path='./Kfold_models/fold{}/model_{}_epoch{}.pkl'.format(fold, fold, epoch))

            if early_stopping.early_stop:
                print("Early stopping at epoch ", epoch)
                break

        np.save('./Kfold_models/fold{}/train_LOSS.npy'.format(fold), np.array(train_LOSS))
        np.save('./Kfold_models/fold{}/train_ACC.npy'.format(fold), np.array(train_ACC))
        np.save('./Kfold_models/fold{}/test_LOSS.npy'.format(fold), np.array(test_LOSS))
        np.save('./Kfold_models/fold{}/test_ACC.npy'.format(fold), np.array(test_ACC))
        np.save('./Kfold_models/fold{}/val_LOSS.npy'.format(fold), np.array(val_LOSS))
        np.save('./Kfold_models/fold{}/val_ACC.npy'.format(fold), np.array(val_ACC))

        del model


if __name__ == '__main__':
    set_random_seed(0)
    train(save_all_checkpoint=False)
