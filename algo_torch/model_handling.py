from uuid import uuid4
import os
import torch
import zipfile
import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def save_model(checkpoint_file, relative_module_path, zip_filename):
    dir_uuid = str(uuid4())
    dir_path = "/tmp/{}/torch_model".format(dir_uuid)
    os.makedirs(dir_path, exist_ok=True)
    os.mkdir("{}/model".format(dir_path))
    os.mkdir("{}/source".format(dir_path))
    model_save_path = "{}/model/model.t7".format(dir_path)
    source_module_path = "{}/source/{}".format(dir_path, relative_module_path.split("/")[-1])

    torch.save(checkpoint_file, model_save_path)
    copytree("{}/{}".format(os.getcwd(), relative_module_path), source_module_path)

    local_zip_path = "/tmp/{}".format(zip_filename)
    with zipfile.ZipFile(local_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(model_save_path, "/".join(model_save_path.split('/')[3:]))
        z.write(source_module_path, "/".join(source_module_path.split('/')[3:]))

    return local_zip_path


def load_model(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        z.extractall("/tmp")
    source_root_dir = "/tmp/torch_model/source"
    source_module_path = "{}/{}".format(source_root_dir, os.listdir(source_root_dir)[0])
    model_path = "/tmp/torch_model/model/model.t7"
    return source_module_path, model_path

