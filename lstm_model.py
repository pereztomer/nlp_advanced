from torch import nn
import torch


class BasicLstm(nn.Module):
    def __init__(self, embedding_dim):
        super(BasicLstm, self).__init__()
        self.embedding_dim = embedding_dim
        hidden_size = 512
        self.encoder = nn.LSTM(input_size=embedding_dim, num_layers=2, bidirectional=True, hidden_size=hidden_size,
                               batch_first=True)
        self.mlp = Mlp(input_dim=hidden_size * 2, output_dim=4)
        self.softmax = nn.Softmax()

    def forward(self, sen_embeddings):
        lstm_out, _ = self.encoder(sen_embeddings)

        prediction = self.mlp(lstm_out[:, -1, :, ])
        return self.softmax(prediction)


class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        return output
