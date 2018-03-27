import torch
import tempfile
import shutil
import sys
import zipfile
def save_model(model, source_path, output_path):
    # data = {'model': model}
    _, model_temp = tempfile.mkstemp()
    _, source_temp = tempfile.mkstemp()
    torch.save(model, model_temp)
    shutil.copy(source_path, source_temp)
    with zipfile.ZipFile(output_path, "w") as zip:
        zip.write(model_temp, "model.t7")
        zip.write(source_temp, source_path)
    return output_path

def load_model(local_file_path):
    temp_loc = '/tmp'
    with zipfile.ZipFile(local_file_path) as zip:
        zip.extractall(temp_loc)
    sys.path.insert(0, temp_loc)
    model = torch.load('{}/model.t7'.format(temp_loc))
    return model
