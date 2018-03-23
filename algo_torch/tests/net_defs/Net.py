import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln1 = torch.nn.Linear(1, 10)
    def forward(self, input):
        output = self.ln1(input)
        return output
