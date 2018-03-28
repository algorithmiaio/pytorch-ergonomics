"""
pytorch is powerful ML library with tons of uses, but the default python wheels are heavy and on algorithmia that heaviness results in slow algorithm execution.
Combining the execute_workaround function with any of our precompiled, slim pytorch wheels can yield a dramatic improvement in performance,
particularly for GPU pytorch algorithms!
"""

import json
import os
import tempfile
from subprocess import PIPE, Popen
from time import sleep
class TorchExecutionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

""" 
This will work as intended for algorithms that follow the Algorithmia standard project hiearchy.
input_data - a json serializable dictionary containing input data for your algorithm
entrypoint_file - the unix path to your main execution script, starting from the project root directory. eg; 'src/runable_example.py'
function - the entry function you want to execute, this function should accept your json dictionary as input, and return an output dictionary.
"""
def execute_workaround(input_data, entrypoint_file, function):
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgfortran.so.3'
    os.environ['LC_ALL'] = 'C'
    os.environ['OMP_NUM_THREADS'] = '1'
    prepare_file(entrypoint_file, function)
    _, in_filename = tempfile.mkstemp()
    _, out_filename = tempfile.mkstemp()
    print(in_filename)
    print(out_filename)
    with open(in_filename, 'w') as f:
        json.dump(input_data, f)
    root_path = os.path.realpath(os.getcwd())
    runShellCommand(['python3', entrypoint_file, in_filename, out_filename], cwd=root_path)
    with open(out_filename) as f:
        data = f.read()
        data = data.replace("\n", "")
        output = json.loads(data)
    return output

def prepare_file(entrypoint_file, function_name):

    with open(entrypoint_file) as f:
        vanilla_text = f.read()
    if """if __name__ == "__main__":""" not in vanilla_text:

        text = """

if __name__ == "__main__":
    import sys
    import json
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    with open(input_filename) as f:
        input = json.loads(f.read())
    output = {}(input)
    with open(output_filename, 'w') as f:
        json.dump(output, f)""".format(function_name)
        with open(entrypoint_file, "a") as f:
            f.write(text)




def runShellCommand(commands, cwd=None):
    p = Popen(commands, stdout=PIPE, stderr=PIPE, cwd=cwd)
    output, error = p.communicate()
    out_str = str(output.decode('utf-8'))
    err_str = str(error.decode('utf-8'))
    print(out_str)
    if "TypeError: \'NoneType\' object is not callable".encode() in error:
        sleep(1)
        return runShellCommand(commands, cwd)
    elif error:
        raise TorchExecutionError(err_str)
    return output