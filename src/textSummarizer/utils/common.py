import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If YAML file is empty or invalid.
        Exception: Any other unforeseen errors.

    Returns:
        ConfigBox: Parsed YAML content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError(f"YAML file {path_to_yaml} is empty")

            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("Invalid YAML content detected")
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create a list of directories.

    Args:
        path_to_directories (list): List of directory paths.
        verbose (bool, optional): Whether to log directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB (approx).
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
