import argparse
import copy
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from dataloader import SuicidalDataset
from model.model import HistoricCurrent, Historic, Current
from utils import pad_ts_collate


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

    return cb_loss


def train_loop(model, dataloader, optimizer, device, dataset_len):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
        labels, tweet_features, temporal_features, lens, timestamp = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)
        temporal_features = temporal_features.to(device)
        lens = lens.to(device)
        timestamp = timestamp.to(device)

        optimizer.zero_grad()
        output = model(tweet_features, temporal_features, lens, timestamp)
        _, preds = torch.max(output, 1)

        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    for bi, inputs in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
        labels, tweet_features, temporal_features, lens, timestamp = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)
        temporal_features = temporal_features.to(device)
        lens = lens.to(device)
        timestamp = timestamp.to(device)

        with torch.no_grad():
            output = model(tweet_features, temporal_features, lens, timestamp)

        _, preds = torch.max(output, 1)
        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)


def loss_fn(output, targets, samples_per_cls):
    beta = 0.9999
    gamma = 2.0
    no_of_classes = 2
    loss_type = "focal"

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def main(config):
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size

    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = config.embedding_dim

    NUM_LAYERS = config.num_layer
    DROPOUT = config.dropout
    CURRENT = config.current

    RANDOM = config.random

    DATA_DIR = config.data_dir

    if config.base_model == "historic":
        model = Historic(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    elif config.base_model == "current":
        model = Current(HIDDEN_DIM, DROPOUT)
    elif config.base_model == "historic-current":
        model = HistoricCurrent(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, config.model)
    else:
        assert False

    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_train = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_val = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_test = pickle.load(f)

    train_dataset = SuicidalDataset(df_train.label.values, df_train.curr_enc.values, df_train.enc.values,
                                    df_train.hist_dates, CURRENT, RANDOM)
    val_dataset = SuicidalDataset(df_val.label.values, df_val.curr_enc.values, df_val.enc.values,
                                  df_val.hist_dates, CURRENT, RANDOM)
    test_dataset = SuicidalDataset(df_test.label.values, df_test.curr_enc.values, df_test.enc.values,
                                   df_test.hist_dates, CURRENT, RANDOM)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_ts_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_ts_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_ts_collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    LEARNING_RATE = config.learning_rate

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=EPOCHS
    )

    model_name = f'{int(datetime.timestamp(datetime.now()))}_{config.base_model}_{config.model}_{config.hidden_dim}_{config.num_layer}_{config.learning_rate}'

    best_metric = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(model)
    print(optimizer)
    print(scheduler)

    for epoch in range(EPOCHS):
        loss, accuracy = train_loop(model, train_dataloader, optimizer, device, len(train_dataset))
        eval_loss, eval_accuracy, __, _ = eval_loop(model, val_dataloader, device, len(val_dataset))

        metric = f1_score(_, __, average="macro")
        recall_1 = recall_score(_, __, average=None)[1]
        if scheduler is not None:
            scheduler.step()

        print(
            f'epoch {epoch + 1}:: train: loss: {loss:.4f}, accuracy: {accuracy:.4f} | valid: loss: {eval_loss:.4f}, accuracy: {eval_accuracy:.4f}, f1: {metric:.4f}, recall_1: {recall_1:.4f}')
        if metric > best_metric:
            best_metric = metric
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 25 == 24:
            if scheduler is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_f1': best_metric
                }, f'{model_name}_{epoch}.tar')

    print(best_metric.item())
    model.load_state_dict(best_model_wts)

    if not os.path.exists('saved_model'):
    	os.mkdir("saved_model")

    torch.save(model.state_dict(), os.path.join(DATA_DIR, f'saved_model/best_model_{model_name}.pt'))

    _, _, y_pred, y_true = eval_loop(model, val_dataloader, device, len(val_dataset))

    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
    print(report)
    result = {'best_f1': best_metric.item(),
              'lr': LEARNING_RATE,
              'model': str(model),
              'optimizer': str(optimizer),
              'scheduler': str(scheduler),
              'base-model': config.base_model,
              'model-name': config.model,
              'epochs': EPOCHS,
              'embedding_dim': EMBEDDING_DIM,
              'hidden_dim': HIDDEN_DIM,
              'num_layers': NUM_LAYERS,
              'dropout': DROPOUT,
              'current': CURRENT,
              'val_report': report}

    with open(os.path.join(DATA_DIR, f'saved_model/VAL_{model_name}.json'), 'w') as f:
        json.dump(result, f)

    if config.test:
        _, _, y_pred, y_true = eval_loop(model, test_dataloader, device, len(test_dataset))

        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
        print(report)
        result['test_report'] = report

        with open(os.path.join(DATA_DIR, f'saved_model/TEST_{model_name}.json'), 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    base_model_set = {"historic", "historic-current", "current"}
    model_set = {"tlstm", "bilstm", "bilstm-attention"}

    parser = argparse.ArgumentParser(description="Temporal Suicidal Modelling")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("-e", "--epochs", default=10, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=100, type=int)
    parser.add_argument("-ed", "--embedding-dim", default=768, type=int)
    parser.add_argument("-n", "--num-layer", default=1, type=int)
    parser.add_argument("-d", "--dropout", default=0.5, type=float)
    parser.add_argument("--current", action="store_false")
    parser.add_argument("--base-model", type=str, choices=base_model_set, default="historic-current")
    parser.add_argument("--model", type=str, choices=model_set, default="tlstm")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--random", action="store_true")
    config = parser.parse_args()

    main(config)
