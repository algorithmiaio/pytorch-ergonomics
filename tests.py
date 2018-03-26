from load_improvement import execute_workaround
import unittest

class ExecutionTest(unittest.TestCase):
    def execute(self):
        input = {'array': [1.2334]}
        entrypoint_file = 'src/runable_example.py'
        entry_function = 'execute_torch'
        output = execute_workaround(input, entrypoint_file, entry_function)
        self.assertTrue(output['result'])