from algo_torch import model_handling
from algo_torch.tests.net_defs import Net
from torch import save
import os
import unittest
class model_tests(unittest.TestCase):
    def test_save(self):
        net = Net.Net()
        path = "/tmp/model.t7"
        save(net, path)
        zip_file_path = model_handling.save_model(path, "net_defs", "torch_model.zip")
        self.assertTrue(os.path.exists(zip_file_path))

