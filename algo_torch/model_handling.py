from uuid import uuid4
import os
from shutil import copytree
import torch
import zipfile

def save_model(checkpoint_file, relative_module_path):
    dir_uuid = str(uuid4())
    dir_path = "/tmp/{}".format(dir_uuid)
    os.mkdir(dir_path)
    model_save_path = "torch_model/model/model.t7".format(dir_path)
    source_module_path = "torch_model/source/{}".format(dir_path, relative_module_path.split("/"[-1]))

    torch.save(checkpoint_file, model_save_path)
    copytree("{}/{}".format(os.getcwd(), relative_module_path), source_module_path)
    local_zip_path = "/tmp/{}.zip".format(dir_uuid)
    with zipfile.ZipFile(local_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(model_save_path)
        z.write(source_module_path)

    return local_zip_path


def load_model(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        z.extractall("/tmp")
    source_root_dir = "/tmp/torch_model/source"
    source_module_path = "{}/{}".format(source_root_dir, os.listdir(source_root_dir)[0])
    model_path = "/tmp/torch_model/model/model.t7"
    return source_module_path, model_path



