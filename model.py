import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


def categorical_accuracy(preds, y):
    """Return accuracy given a batch of label distributions and true labels"""
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum().float() / torch.FloatTensor([y.shape[0]])


class LR(nn.Module):
    """Logistic regressor over mean input embedding"""

    def __init__(self, vocab_size, embedding_dim, n_classes):
        super(LR, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, data, probs=False):
        text, length = data
        text_embed = self.embeddings(text)

        mean_embed = text_embed.sum(0)
        mean_embed /= (length.float().unsqueeze(1) + 1)

        logits = self.classifier(mean_embed)
        return logits


class DAN(nn.Module):
    """Deep Averaging Network (Iyyer et al., 2015)."""

    def __init__(self, vocab_size, embedding_dim, n_classes, n_layers=3,
                 hidden_dim=300, dropout=0.0):
        super(DAN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        layers = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
                  nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.classifier = nn.Sequential(*layers)
        self._softmax = nn.LogSoftmax()

    def forward(self, data, probs=False):
        # data is (text, length) tuple
        # text.shape == (sent len, batch size)
        text, length = data

        # text_embed.shape == (sent len, batch size, emb dim)
        text_embed = self.embeddings(text)

        # mean_embed.shape == (batch size, emb dim)
        mean_embed = text_embed.sum(0)
        mean_embed /= (length.float().unsqueeze(1) + 1)

        logits = self.classifier(mean_embed)
        if probs:
            return self._softmax(logits)
        else:
            return logits


class CNN(nn.Module):
    """Convolutional neural networks from (Kim, 2014)."""

    def __init__(self, vocab_size, embedding_dim, n_classes,
                 n_filters=100, filter_sizes=(3, 4, 5), dropout=.0):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes])
        self.classifier = nn.Linear(len(filter_sizes)*n_filters, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        # text.shape == [sent len, batch size]
        text, length = batch

        x = text.permute(1, 0)
        # x.shape = [batch size, sent len]
        embedded = self.embeddings(x)
        # embedded.shape = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded.shape == [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved[n].shape = =[batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled.shape == [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(cat)


def get_model(model_type, vocab_size, emb_dim, n_classes, dropout=.0):
    if model_type == 'lr':
        if dropout > 0:
            logging.warning("Logistic Regression doesn't support dropout.")
        return LR(vocab_size, emb_dim, n_classes)
    elif model_type == 'dan':
        return DAN(vocab_size, emb_dim, n_classes, dropout=dropout)
    elif model_type == 'cnn':
        return CNN(vocab_size, emb_dim, n_classes, dropout=dropout)
    else:
        raise ValueError('Model type not implemented')


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_rank(model, iterator, criterion):
    model.embeddings.weight.requires_grad = True
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    for batch in iterator:
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        loss.backward()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)