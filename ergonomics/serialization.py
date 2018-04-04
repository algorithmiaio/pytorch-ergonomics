"""
These functions are ergonomics improvements for pytorch.
   By saving the source code along with the model definitions,
   you can reuse your models easily in new modules without needing to redefine
   everything from scratch.
"""

import torch
import tempfile
import os
import zipfile
import sys
import pickle
DEFAULT_MOD_PATH = 'portable-pytorch'

def save_portable(obj: torch.nn.Module, mod_path: str, output_path: str, path_delimiter='/'):
    """
    Portably serializes a torch module file to a disk, along with the network definition module required to execute it.
    Args:
        obj: the torch module to save.
        mod_path: the pythonic import path to your network definition module (it's recommended that this is
        a separate module, that is self contained (IE does not use resources from other parts of your project).
        output_path: the output file path where you'd like to save your model to.
        path_delimiter: how your operating system delimits filepaths, the default assumes you're using a unix kernel.
    Example:
         # save portable version of your model
         from src.some_project.pytorch_defs.net import Net
         net = Net(...)
         output = net.forward()
         ...
         torch.save_portable(net, "src.some_project.pytorch_defs", "/tmp/myModel.zip")
    """
    _, model_temp = tempfile.mkstemp()
    input_system_path = mod_path.replace('.', path_delimiter)
    _save_portable(obj, mod_path, model_temp)
    source_files = []
    for root, dirs, files in os.walk(input_system_path):
        for file in files:
            true_path = os.path.join(root, file)
            false_path = os.path.join(DEFAULT_MOD_PATH, file)
            source_files.append((true_path, false_path))
    with zipfile.ZipFile(output_path, "w") as zip:
        zip.write(model_temp, "model.t7")
        for true_path, false_path in source_files:
            zip.write(true_path, false_path)
    os.remove(model_temp)
    return output_path


def _save_portable(obj: torch.nn.Module, input_mod_path: str, model_save_path: str):
    serialized_model = pickle.dumps(obj)
    input_mod_path = input_mod_path.encode('utf-8')
    output_mod_path = DEFAULT_MOD_PATH.encode('utf-8')
    serialized_model = serialized_model.replace(input_mod_path, output_mod_path)
    with open(model_save_path, 'wb') as f:
        f.write(serialized_model)

def load_portable(local_file_path, temp_location="/tmp", path_delimiter='/'):
    """
    Portably deserializes a portable serialized torch model saved with "save_portable". The magic here is
    you don't need the original network definition code in order to load this network and utilize it, as the
    necessary source code is provided along with the network.
    Args:
        local_file_path: The local system file path to your portable model object.
        temp_location: a scratchspace location to use for the unzip/depickling process, if you're
        using a non debian based kernel (or wholy different operating system) you'll want to change this.
        path_delimiter: how your operating system delimits filepaths, the default assumes you're using a unix kernel.
    Example:
         # Load a portable model
         local_portable_model_path = "/tmp/myModel.zip"
         model = torch.load_portable(local_portable_model_path)
         output = model.forward(...)
    """
    with zipfile.ZipFile(local_file_path) as zip:
        zip.extractall(temp_location)
    sys.path.insert(0, temp_location)
    model = _load_portable("{}{}model.t7".format(str(temp_location), path_delimiter))
    return model


def _load_portable(model_path):
    with open(model_path, 'rb') as f:
        mod = pickle.load(f)
    return mod