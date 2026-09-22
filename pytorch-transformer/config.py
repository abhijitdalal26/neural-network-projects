from pathlib import Path, PurePosixPath

def _model_dir(config):
    # Public repo is Helsinki-NLP/opus_books; keep local folder as opus_books_weights for backward compat
    # and respect absolute Drive paths (e.g. /content/drive/... on Linux/Colab)
    mf = config['model_folder']
    # POSIX absolute check via string (Path.is_absolute is OS-dependent and fails for /content on Windows)
    if mf.startswith("/") or mf.startswith("\\") or Path(mf).is_absolute():
        return mf
    ds_base = config['datasource'].split("/")[-1]
    return f"{ds_base}_{mf}"

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "nl",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def _join_model_path(model_folder: str, filename: str) -> str:
    # Colab Drive paths are POSIX even when tested on Windows — preserve leading /
    if model_folder.startswith("/"):
        return str(PurePosixPath(model_folder) / filename)
    return str(Path(model_folder) / filename)

def get_weights_file_path(config, epoch: str):
    model_folder = _model_dir(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return _join_model_path(model_folder, model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = _model_dir(config)
    model_filename = f"{config['model_basename']}*"
    # Glob needs OS-correct Path; for POSIX Drive path on Linux this is fine
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])