import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, relu


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointerNetwork(nn.Module):
    def __init__(self, input_size=5, output_size=5, embedding_size=256, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encoder_embedding = nn.Linear(1, embedding_size)
        self.decoder_embedding = nn.Embedding(output_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size)
        self.decoder = nn.GRUCell(embedding_size, hidden_size)
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, inputs, targets, training=True):
        batch_size = inputs.size(1)
        max_len = targets.size(0)
        embedded_inputs = torch.zeros(
            (self.input_size, batch_size, self.embedding_size)).to(device)
        for i in range(batch_size):
            for j in range(self.input_size):
                embedded_inputs[j][i] = self.encoder_embedding(inputs[j][i].resize_(1, 1).float())
        if training:
            targets = self.decoder_embedding(targets)
        encoder_outputs, hidden = self.encoder(embedded_inputs)
        decoder_outputs = torch.zeros((max_len, batch_size, self.output_size)).to(device)
        decoder_input = torch.zeros((batch_size, self.embedding_size)).to(device)
        hidden = hidden.squeeze(0)  # (B, H)
        for i in range(max_len):
            hidden = self.decoder(decoder_input, hidden)
            projection1 = self.w1(encoder_outputs)
            projection2 = self.w2(hidden)
            output = log_softmax(self.v(
                relu(projection1 + projection2)).squeeze(-1).transpose(0, 1), -1)
            decoder_outputs[i] = output
            if training:
                decoder_input = targets[i]
            else:
                _, indices = torch.max(output, 1)
                decoder_input = self.decoder_embedding(indices)
        if training:
            return decoder_outputs
        else:
            outputs = torch.zeros(batch_size, self.output_size).to(device)
            decoder_outputs = decoder_outputs.permute(1, 0, 2)
            for i in range(batch_size):
                outputs[i] = torch.argmax(decoder_outputs[i], 1)
            return outputs


class PtrNet(nn.Module):
    def __init__(self, input_size=5, output_size=5, embedding_size=256, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encoder_embedding = nn.Embedding(101, embedding_size)
        self.decoder_embedding = nn.Embedding(output_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size)
        self.decoder = nn.GRUCell(embedding_size, hidden_size)
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, inputs, targets, training=True):
        batch_size = inputs.size(1)
        max_len = targets.size(0)
        embedded_inputs = self.encoder_embedding(inputs)
        if training:
            targets = self.decoder_embedding(targets)
        encoder_outputs, hidden = self.encoder(embedded_inputs)
        decoder_outputs = torch.zeros((max_len, batch_size, self.output_size)).to(device)
        decoder_input = torch.zeros((batch_size, self.embedding_size)).to(device)
        hidden = hidden.squeeze(0)  # (B, H)
        for i in range(max_len):
            hidden = self.decoder(decoder_input, hidden)
            projection1 = self.w1(encoder_outputs)
            projection2 = self.w2(hidden)
            output = log_softmax(self.v(
                relu(projection1 + projection2)).squeeze(-1).transpose(0, 1), -1)
            decoder_outputs[i] = output
            if training:
                decoder_input = targets[i]
            else:
                _, indices = torch.max(output, 1)
                decoder_input = self.decoder_embedding(indices)
        if training:
            return decoder_outputs
        else:
            outputs = torch.zeros(batch_size, self.output_size).to(device)
            decoder_outputs = decoder_outputs.permute(1, 0, 2)
            for i in range(batch_size):
                outputs[i] = torch.argmax(decoder_outputs[i], 1)
            return outputs
