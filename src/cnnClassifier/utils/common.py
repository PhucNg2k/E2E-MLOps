from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
from cnnClassifier import logger
from box import ConfigBox
from box.exceptions import BoxValueError
import os
import base64
import yaml
import json
import joblib


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except ValueError as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dumps(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(data: Any, path: Path):
    """
    Save binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(filename=path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round( os.path.getsize(path) / 1024 )
    return f"~{size_in_kb} KB"

def decodeImage(imgString, fileName):
    imgdata = base64.b64decode(imgString)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())



