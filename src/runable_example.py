import torch

class PredefNet(torch.nn.Module):
    def __init__(self):
        super(PredefNet, self).__init__()
        self.ln1 = torch.nn.Linear(1, 5)
    def forward(self, input):
        output = self.ln1(input)
        return output

def execute_torch(input_data):
    data = input_data['array']
    tensor = torch.autograd.Variable(torch.FloatTensor(data), requires_grad=False)
    net = PredefNet()
    results = net.forward(tensor)
    output_array = []
    for result in list(results.cpu().data.numpy()):
        output_array.append(float(result))
    output = {'result': output_array}
    return output
