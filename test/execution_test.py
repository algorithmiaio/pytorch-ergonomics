import sys
sys.path.append('../')
import unittest
execution_script = "ergonomics/src/runable_example.py"


class ExecutionTest(unittest.TestCase):
    def test_workaround(self):
        from ergonomics.algorithmia import execute_workaround
        input = {'array': [1.2334]}
        entrypoint_file = execution_script
        entry_function = 'execute_torch'
        output = execute_workaround(input, entrypoint_file, entry_function)
        self.assertTrue(output['result'])


if __name__ =="__main__":
    unittest.main()