from abc import ABC, abstractmethod

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from preprocess import factory
from inference import Inference
from utils import get_logger, log_experiment_mlflow, recursive_config_update


class ExperimentRunner(ABC):

    @staticmethod
    @abstractmethod
    def run_experiment(cfg: DictConfig):
        pass


class EmbedText(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute embeddings or skip if already computed
        image_ids, embeddings, model = Inference.compute_text_embeddings(
            cfg, df.file_name.tolist(), df.caption.tolist()
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        # create mlflow experiment + log metrics
        log_experiment_mlflow(
            cfg, results, [model], [cfg.model.sentence_transformer.model_name],
            ['pytorch_image_model', 'image_captioning', 'open_clip']
        )


class EmbedImage(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute embeddings or skip if already computed
        image_ids, embeddings, model = Inference.compute_image_embeddings(
            cfg, df.file_name.tolist(), df.image_path.tolist()
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        # create mlflow experiment + log metrics
        log_experiment_mlflow(
            cfg, results, [model], [cfg.model.pytorch_image_model.model_name],
            ['sentence_transformer', 'image_captioning', 'open_clip']
        )


class EmbedTextImage(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute embeddings or skip if already computed
        image_ids, text_embeddings, text_model = Inference.compute_text_embeddings(
            cfg, df.file_name.tolist(), df.caption.tolist()
        )
        _, image_embeddings, image_model = Inference.compute_image_embeddings(
            cfg, df.file_name.tolist(), df.image_path.tolist()
        )
        # concatenate text + image embeddings
        embeddings = np.concatenate(
            (text_embeddings, image_embeddings), axis=1
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        # create mlflow experiment + log metrics
        models = [text_model, image_model]
        model_names = [
            cfg.model.sentence_transformer.model_name,
            cfg.model.pytorch_image_model.model_name
        ]
        log_experiment_mlflow(
            cfg, results, models, model_names,
            ['image_captioning', 'open_clip']
        )


class MultiModalVitEmbedImage(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute embeddings or skip if already computed
        image_ids, embeddings, model = Inference.compute_image_embeddings_open_clip(
            cfg, df.file_name.tolist(), df.image_path.tolist()
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        mn, pt = cfg.model.open_clip.model_name, cfg.model.open_clip.pretrained
        # create mlflow experiment + log metrics
        log_experiment_mlflow(
            cfg, results, [model], [f'{mn}_{pt}'],
            ['sentence_transformer', 'pytorch_image_model', 'image_captioning']
        )


class MultiModalVitEmbedImageText(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute embeddings or skip if already computed
        image_ids, embeddings, model = Inference.compute_text_image_embeddings_open_clip(
            cfg, df.file_name.tolist(), df.caption.tolist(),
            df.image_path.tolist()
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        mn, pt = cfg.model.open_clip.model_name, cfg.model.open_clip.pretrained
        # create mlflow experiment + log metrics
        log_experiment_mlflow(
            cfg, results, [model], [f'{mn}_{pt}'],
            ['sentence_transformer', 'pytorch_image_model', 'image_captioning']
        )


class ConcatSentenceTransformerOpenCLIP:
    """Computes text embeddings with a Sentence Transformer and image
    embeddings with an OpenCLIP model to concatenate the results."""
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute text and embeddings or skip if already computed
        image_ids, text_embeddings, text_model = Inference.compute_text_embeddings(
            cfg, df.file_name.tolist(), df.caption.tolist()
        )
        _, image_embeddings, image_model = Inference.compute_image_embeddings_open_clip(
            cfg, df.file_name.tolist(), df.image_path.tolist()
        )
        # concatenate text + image embeddings
        embeddings = np.concatenate(
            (text_embeddings, image_embeddings), axis=1
        )
        # run evaluation (0th index is the vector itself + 1)
        top_k = cfg.top_k + 1
        results = Inference.evaluate_embeddings(
            df, image_ids, embeddings, top_k
        )
        # create mlflow experiment + log metrics
        models = [text_model, image_model]
        model_names = [
            cfg.model.sentence_transformer.model_name,
            cfg.model.pytorch_image_model.model_name
        ]
        log_experiment_mlflow(
            cfg, results, models, model_names,
            ['image_captioning', 'open_clip']
        )


class CaptionImagesBlip(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        # preprocess
        df = factory(cfg).preprocess(cfg)
        # compute image captions
        _ = Inference.compute_image_captions_blip(
            cfg, df.file_name.tolist(), df.image_path.tolist()
        )


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg: DictConfig) -> None:
    def wrap_experiment(cfg: DictConfig, exp_name: str) -> None:

        if exp_name in globals() and callable(globals()[exp_name]):
            experiment_runner: ExperimentRunner = globals()[exp_name]
            experiment_runner.run_experiment(cfg)
        else:
            raise NotImplementedError(
                f'Configured experiment class {exp_name} is not found!'
            )

    logger = get_logger()

    if cfg.pipeline:
        logger.info(f'Running {len(cfg.pipeline)} experiments serially.')
        for exp_params in tqdm(cfg.pipeline):
            cfg_exp = cfg.copy()
            dict_exp = OmegaConf.to_container(cfg_exp, resolve=True)
            dict_update = OmegaConf.to_container(exp_params, resolve=True)
            recursive_config_update(dict_exp, dict_update)
            cfg_exp = OmegaConf.create(dict_exp)
            exp_name = cfg_exp.mlflow.experiment_name
            logger.info(
                f'Running {exp_name} experiment with parameters: {exp_params}'
            )
            wrap_experiment(cfg_exp, exp_name)
    else:
        logger.info('Running experiment with the main config.')
        exp_name = cfg.mlflow.experiment_name
        wrap_experiment(cfg, exp_name)


if __name__ == '__main__':

    main()
