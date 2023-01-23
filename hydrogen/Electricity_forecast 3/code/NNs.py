import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim * 24 * 4, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.double()
        h0 = torch.randn(self.num_layers, x.shape[0], self.hidden_dim).to(self.device).double()
        c0 = torch.randn(self.num_layers, x.shape[0], self.hidden_dim).to(self.device).double()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.relu(out)
        return out

    def predict(self, x):
        return self.forward(x).detach()


def train(model, train_dl, num_epochs, opt, loss_fn):
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        actual_loss = 0
        for i, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device).double()
            pred = model(xb)
            opt.zero_grad()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            actual_loss += loss.detach().item()

        losses.append(actual_loss / (i + 1))
        print(f'Epoch: {epoch} | Loss: {actual_loss / (i + 1)}')
    return losses
