import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from nlp_advanced_utils import CustomDatasetAug
from torch.utils.data import DataLoader
from lstm_model import BasicLstm
from torch import nn


def train(model, train_data_loader, validation_data_loader, epochs, loss_func, lr, device):
    print('Beginning training')

    model = model.to(device)
    train_loss_list = []
    val_loss_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        model.train()
        epoch_train_loss = 0
        for train_sentences_batch, train_labels_batch in train_data_loader:
            optimizer.zero_grad()

            train_sentences_batch = train_sentences_batch.to(device)
            train_labels_batch = train_labels_batch.to(device).type(torch.float32)

            model_predictions = model(train_sentences_batch)
            model_predictions = torch.squeeze(model_predictions)
            batch_loss = loss_func(train_labels_batch, model_predictions)

            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()

        train_loss_list.append(epoch_train_loss / len(train_data_loader))

        # evaluating
        model.eval()
        epoch_val_loss = 0
        for val_sentences_batch, val_labels_batch in validation_data_loader:
            val_sentences_batch = val_sentences_batch.to(device)
            val_labels_batch = val_labels_batch.to(device)

            model_predictions = model(val_sentences_batch)
            model_predictions = torch.squeeze(model_predictions)

            batch_loss = loss_func(val_labels_batch, model_predictions)

            epoch_val_loss += batch_loss.item()

        val_loss_list.append(epoch_val_loss / len(validation_data_loader))

        print(f'Epoch: {i}, train loss: {train_loss_list[-1]}, validation loss: {val_loss_list[-1]}')

    plot_graph(train_loss_list, val_loss_list, graph_type='cross entropy loss')


def plot_graph(train_loss, val_loss, graph_type):
    plt.plot(train_loss, label=f'train {graph_type}')
    plt.plot(val_loss, label=f'validation {graph_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'basic model {graph_type} - train/val')
    plt.legend()
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    json_path = 'data.json'
    epochs = 100
    lr = 0.001
    model = BasicLstm(embedding_dim=200)
    ds = DataLoader(CustomDatasetAug(json_path=json_path, seq_len=400),
                    batch_size=10,
                    shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    train(model=model,
          train_data_loader=ds,
          validation_data_loader=ds,
          epochs=epochs,
          lr=lr,
          loss_func=loss_func,
          device=device)


if __name__ == '__main__':
    main()
