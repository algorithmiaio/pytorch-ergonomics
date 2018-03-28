import unittest

source_file = "src"
model_path = "/tmp/saved_model_and_source.zip"


class ExecutionTest(unittest.TestCase):
    def execute(self):
        from algorithm_ergonomics import execute_workaround
        input = {'array': [1.2334]}
        entrypoint_file = 'src/runable_example.py'
        entry_function = 'execute_torch'
        output = execute_workaround(input, entrypoint_file, entry_function)
        self.assertTrue(output['result'])

class SavingTest(unittest.TestCase):
    def execute(self):
        from src.runable_example import PredefNet
        from model_ergonomics import save_model
        import os
        import torch
        net = PredefNet()
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        _ = net.forward(data)
        output_path = save_model(net, source_file, model_path)
        self.assertTrue(os.path.exists(output_path))

class LoadingTest(unittest.TestCase):
    def execute(self):
        from model_ergonomics import load_model
        import torch
        network = load_model(model_path)
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        output = network.forward(data)
        print(output.data)
        self.assertGreaterEqual(len(list(output.data)), 0)