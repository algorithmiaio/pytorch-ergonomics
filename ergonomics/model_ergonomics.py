"""
These functions are ergonomics improvements for pytorch.
   By saving the source code along with the model definitions,
   you can reuse your models easily in new modules without needing to redefine
   everything from scratch.
"""

import torch
from . import torch_model_ops
import tempfile
import os
import zipfile
import sys
"""
Saves a model file along with it's source module.
If your definition is not in a separate module, passing `src` should work in a typical
Algorithmia pythonic directory stucture.
model - the pytorch network module object that you wish to save
input_mod_path - the pythonic module path to your network definition module.
output_path - the local system filename you'd like to save your network module as.

"""

default_mod_path = 'ergo-pytorch'

def save_model(model: torch.nn.Module, input_mod_path: str, output_path: str):
    _, model_temp = tempfile.mkstemp()
    input_system_path = input_mod_path.replace('.', '/')
    torch_model_ops.save(model, input_mod_path, default_mod_path, model_temp)
    source_files = []
    for root, dirs, files in os.walk(input_system_path):
        for file in files:
            true_path = os.path.join(root, file)
            false_path = os.path.join(default_mod_path, file)
            source_files.append((true_path, false_path))
    with zipfile.ZipFile(output_path, "w") as zip:
        zip.write(model_temp, "model.t7")
        for true_path, false_path in source_files:
            zip.write(true_path, false_path)
    os.remove(model_temp)
    return output_path

"""
Loads a model and source definitions that were saved using the `save_model` function above.
assumes that you're using linux, and that `/tmp` is an available working directory.
local_file_path - the path to the zipped model & source definitions on your local system.
"""


def load_model(local_file_path: str, temp_location: str):
    with zipfile.ZipFile(local_file_path) as zip:
        zip.extractall(temp_location)
    sys.path.insert(0, temp_location)
    model = torch_model_ops.load("{}/model.t7".format(str(temp_location)))
    return model