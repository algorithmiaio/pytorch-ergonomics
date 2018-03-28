import unittest

source_module = "ergonomics/src"
execution_script = "ergonomics/src/runable_example.py"
model_path = "/tmp/saved_model_and_source.zip"
temp_directory = "/tmp"

class ExecutionTest(unittest.TestCase):
    def execute(self):
        from ergonomics.algorithm_ergonomics import execute_workaround
        input = {'array': [1.2334]}
        entrypoint_file = execution_script
        entry_function = 'execute_torch'
        output = execute_workaround(input, entrypoint_file, entry_function)
        self.assertTrue(output['result'])

class SavingTest(unittest.TestCase):
    def execute(self):
        from ergonomics.src.runable_example import PredefNet
        from ergonomics.model_ergonomics import save_model
        import os
        import torch
        net = PredefNet()
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        _ = net.forward(data)
        output_path = save_model(net, source_module, model_path)
        self.assertTrue(os.path.exists(output_path))

class LoadingTest(unittest.TestCase):
    def execute(self):
        from ergonomics.model_ergonomics import load_model
        import torch
        network = load_model(model_path, temp_directory)
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        output = network.forward(data)
        print(output.data)
        self.assertGreaterEqual(len(list(output.data)), 0)