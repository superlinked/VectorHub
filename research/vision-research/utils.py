import inspect
import json
import logging
from builtins import bool
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch.nn
from omegaconf import DictConfig, OmegaConf
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer


def read_json(path_json: Path) -> dict:

    with open(path_json, 'r') as file:
        data = json.load(file)

    return data


def get_logger() -> logging.Logger:
    caller = inspect.stack()[1][3]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(caller)


def save_embeddings(
    path_save: Path,
    image_ids: List[str],
    embeddings: np.ndarray
) -> None:

    if not path_save.is_dir():
        path_save.mkdir(parents=True)
    else:
        get_logger().warning(
            'Embeddings directory exists, it may happen that you are ' 
            'overwriting previous results...'
        )
    with open(path_save.joinpath('image_ids.json'), 'w') as file:
        json.dump({'image_ids': image_ids}, file)

    with open(path_save.joinpath('embeddings.npy'), 'wb') as file:
        np.save(file, embeddings)


def load_embeddings(path_save: Path) -> Tuple[List[str], np.ndarray]:

    with open(path_save.joinpath('image_ids.json'), 'r') as file:
        image_ids = json.load(file)['image_ids']

    with open(path_save.joinpath('embeddings.npy'), 'rb') as file:
        embeddings = np.load(file)

    return image_ids, embeddings


def check_if_embeddings_exist(path_save: Path) -> bool:

    return path_save.joinpath('embeddings.npy').is_file()


def save_captions(
    path_save: Path,
    file_name: str,
    df: pd.DataFrame
) -> None:

    if not path_save.is_dir():
        path_save.mkdir(parents=True)
    else:
        get_logger().warning(
            'Captions directory exists, it may happen that you are '
            'overwriting previous results...'
        )
    df.to_csv(path_save.joinpath(file_name), index=False)


def log_experiment_mlflow(
    config: DictConfig,
    metrics: Dict[str, float],
    models: List[Union[torch.nn.Module, SentenceTransformer]],
    model_names: List[str],
    model_config_keys: List[str]
) -> None:

    logger = get_logger()
    logger.info('Logging results with MLFlow...')
    config = select_relevant_config(config, model_config_keys)
    cfg_mlflow = config.mlflow
    mlflow.set_tracking_uri(Path().resolve().joinpath(config.experiments))
    mlflow.set_experiment(cfg_mlflow.experiment_name)

    params_to_log = {}

    for i, (model, model_name) in enumerate(zip(models, model_names)):

        n_params = sum(p.numel() for p in model.parameters())
        params_to_log.update({
            f'model_{i}_n_params': f'{n_params:,}',
            f'model_{i}_name': model_name,
            'top_k': config.top_k
        })
        if hasattr(model, 'default_cfg'):
            params_to_log.update(model.default_cfg)

    with mlflow.start_run(
        run_name=cfg_mlflow.run_name,
        description=cfg_mlflow.description
    ) as run:

        mlflow.set_tags(cfg_mlflow.tags)
        mlflow.log_dict(OmegaConf.to_container(config), 'config.yaml')
        mlflow.log_metrics(metrics)
        mlflow.log_params(params_to_log)


def select_relevant_config(
    config: DictConfig,
    model_config_keys: List[str]
) -> DictConfig:
    # remove irrelevant part related to an experiment
    config._set_flag("struct", False)

    if 'pipeline' in config.keys():
        del config['pipeline']

    for key in model_config_keys:
        if key in config.model.keys():
            del config.model[key]

    config._set_flag("struct", True)

    return config


def recursive_config_update(
    cfg_original: dict,
    cfg_update: dict
) -> dict:
    """"""
    for key, value in cfg_update.items():
        if (
                isinstance(value, dict)
                and key in cfg_original
                and isinstance(cfg_original[key], dict)
        ):
            recursive_config_update(cfg_original[key], value)
        else:
            cfg_original[key] = value
