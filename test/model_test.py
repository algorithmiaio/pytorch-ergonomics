import sys
sys.path.append('../')
import unittest
mod_path = "ergonomics.src"
model_path = "/tmp/saved_model_and_source.t7"
temp_directory = "/tmp"


class ModelTest(unittest.TestCase):
    def test_a_saving(self):
        from ergonomics.src.runable_example import PredefNet
        from ergonomics.model_ergonomics import save_model
        import os
        import torch
        net = PredefNet()
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        _ = net.forward(data)
        output_path = save_model(net, mod_path, model_path)
        self.assertTrue(os.path.exists(output_path))

    def test_b_loading(self):
        from ergonomics.model_ergonomics import load_model
        import torch
        network = load_model(model_path, temp_directory)
        data = torch.autograd.Variable(torch.FloatTensor([1.25]))
        output = network.forward(data)
        print(output.data)
        self.assertGreaterEqual(len(list(output.data)), 0)

if __name__ =="__main__":
    unittest.main()