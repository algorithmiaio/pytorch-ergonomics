"""
These functions are ergonomics improvements for pytorch.
   By saving the source code along with the model definitions,
   you can reuse your models easily in new modules without needing to redefine
   everything from scratch.
   """
import torch
import tempfile
import sys
import os
import zipfile
"""
Saves a model file along with it's source module.
If your definition is not in a separate module, passing `src` should work in a typical
Algorithmia pythonic directory stucture.
model - the pytorch network module object that you wish to save
source path - the relative path to the module containing your network definition
output_path - the local system filename you'd like to save your network module as.

"""
def save_model(model, source_path, output_path):
    # data = {'model': model}
    _, model_temp = tempfile.mkstemp()
    torch.save(model, model_temp)
    source_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            path = os.path.join(root, file)
            source_files.append(path)
    with zipfile.ZipFile(output_path, "w") as zip:
        zip.write(model_temp, "model.t7")
        for file in source_files:
            zip.write(file)
    os.remove(model_temp)
    return output_path

"""
Loads a model and source definitions that were saved using the `save_model` function above.
assumes that you're using linux, and that `/tmp` is an available working directory.

local_file_path - the path to the zipped model & source definitions on your local system.
"""


def load_model(local_file_path):
    temp_loc = '/tmp'
    with zipfile.ZipFile(local_file_path) as zip:
        zip.extractall(temp_loc)
    sys.path.insert(0, temp_loc)
    model = torch.load('{}/model.t7'.format(temp_loc))
    return model
