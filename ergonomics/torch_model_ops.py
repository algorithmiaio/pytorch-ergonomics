import torch
import dill


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def save(obj: torch.nn.Module, original_mod_path: str, output_mod_path: str, filepath: str):
    filepath = _save(obj, original_mod_path, output_mod_path, filepath)
    return filepath

def load(filepath: str):
    mod = _load(filepath)
    return mod

def _save(obj, input_mod_path, output_mod_path, filepath):
    serialized_model = dill.dumps(obj, byref=False)
    input_mod_path = input_mod_path.encode('utf-8')
    output_mod_path = output_mod_path.encode('utf-8')
    serialized_model = serialized_model.replace(input_mod_path, output_mod_path)
    with open(filepath, 'wb') as f:
        f.write(serialized_model)
    return filepath



def _load(filepath):
    with open(filepath, 'rb') as f:
        mod = dill.load(f)
    return mod