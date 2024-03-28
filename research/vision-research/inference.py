from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import open_clip
import torch
import torchmetrics
import timm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor, BlipModel, BlipProcessor, BlipForConditionalGeneration
)
from tqdm import tqdm

from preprocess import DataLoaderCollection
from utils import (
    check_if_embeddings_exist,
    get_logger,
    load_embeddings,
    save_embeddings,
    save_captions
)


class Inference:
    @staticmethod
    def compute_text_embeddings(
        config: DictConfig,
        image_ids: List[str],
        captions: List[str]
    ) -> Tuple[List[str], np.ndarray, SentenceTransformer]:

        logger = get_logger()
        cm = config.model.sentence_transformer
        model_name = cm.model_name
        path_save = Path().resolve().joinpath(
            config.text_embeddings, model_name + cm.path_suffix
        )
        if "Salesforce/blip" in model_name:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            model = BlipModel.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)

            if check_if_embeddings_exist(path_save):
                logger.info(
                    'Text embeddings are already computed with the same '
                    'config! Reading and returning already computed '
                    'embeddings.'
                )
                image_ids, embeddings = load_embeddings(path_save)

                return image_ids, embeddings, model

            model = model.to(device).eval()
            logger.info(
                f'Computing {len(captions):,} text embeddings with {model_name}'
            )
            embeddings = []

            bs = cm.encode.batch_size
            batches = [captions[i: i + bs] for i in range(0, len(captions), bs)]

            for text_batch in tqdm(batches):
                inputs = processor(
                    text=text_batch, padding=True, return_tensors="pt"
                ).to(device)
                embedding_batch = model.get_text_features(
                    **inputs, return_dict=False
                )
                embedding_batch = embedding_batch.cpu().detach().numpy()
                embeddings.append(embedding_batch)

            embeddings = np.concatenate(embeddings, axis=0)
        else:
            model = SentenceTransformer(model_name)

            if check_if_embeddings_exist(path_save):
                logger.info(
                    'Text embeddings are already computed with the same '
                    'config! Reading and returning already computed '
                    'embeddings.'
                )
                image_ids, embeddings = load_embeddings(path_save)

                return image_ids, embeddings, model

            logger.info(
                f'Computing {len(captions):,} text embeddings with {model_name}'
            )
            embeddings = model.encode(captions, **cm.encode)

        if cm.save_embeddings:
            save_embeddings(path_save, image_ids, embeddings)

        return image_ids, embeddings, model

    @staticmethod
    def compute_image_embeddings(
        config: DictConfig,
        image_ids: List[str],
        image_paths: List[Path]
    ) -> Tuple[List[str], np.ndarray, torch.nn.Module]:

        logger = get_logger()
        cm = config.model.pytorch_image_model
        model_name = cm.model_name
        path_save = Path().resolve().joinpath(
            config.image_embeddings, model_name + cm.path_suffix
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if "Salesforce/blip" in model_name:
            model = BlipModel.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)

            dataloader = DataLoaderCollection.get_blip_image_dataloader(
                image_paths, processor, cm.batch_size, cm.n_workers
            )
            if check_if_embeddings_exist(path_save):
                logger.info(
                    'Image embeddings are already computed with the same '
                    'config! Reading and returning already computed '
                    'embeddings.'
                )
                image_ids, embeddings = load_embeddings(path_save)

                return image_ids, embeddings, model

            model = model.to(device).eval()
            logger.info(
                f'Computing {len(image_paths):,} image embeddings with '
                f'{model_name}'
            )
            embeddings = []

            for image_batch in tqdm(dataloader):
                image_batch = image_batch.to(device)
                embedding_batch = model.get_image_features(image_batch)
                embedding_batch = embedding_batch.cpu().detach().numpy()
                embeddings.append(embedding_batch)
        else:
            model = timm.create_model(model_name, pretrained=True).to(device)

            if check_if_embeddings_exist(path_save):
                logger.info(
                    'Image embeddings are already computed with the same '
                    'config! Reading and returning already computed '
                    'embeddings.'
                )
                image_ids, embeddings = load_embeddings(path_save)

                return image_ids, embeddings, model

            data_loader = DataLoaderCollection.get_timm_image_dataloader(
                image_paths, model, cm.batch_size, cm.n_workers
            )
            logger.info(
                f'Computing {len(image_paths):,} image embeddings with '
                f'{model_name}'
            )
            embeddings = []

            for image_batch in tqdm(data_loader):
                embedding_batch = model(image_batch.to(device))
                embedding_batch = embedding_batch.cpu().detach().numpy()
                embeddings.append(embedding_batch)

        embeddings = np.concatenate(embeddings, axis=0)

        if cm.save_embeddings:
            save_embeddings(path_save, image_ids, embeddings)

        return image_ids, embeddings, model

    @staticmethod
    def compute_image_captions_blip(
        config: DictConfig,
        image_ids: List[str],
        image_paths: List[Path]
    ) -> pd.DataFrame:
        """"""
        logger = get_logger()
        cm = config.model.image_captioning
        model_name = cm.model_name
        file_name = cm.file_name
        path_save = Path().resolve().joinpath(
            config.image_captions, model_name + cm.path_suffix
        )
        if path_save.is_dir():
            logger.info(
                'Image captions are already computed with the same config! '
                'Reading and returning already computed captions.'
            )
            df = pd.read_csv(path_save.joinpath(file_name))

            return df

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(
            model_name).to(device)
        logger.info(
            f'Computing {len(image_paths):,} unconditional image captions '
            f'with {model_name}'
        )
        captions = []

        dataloader = DataLoaderCollection.get_blip_image_dataloader(
            image_paths, processor, cm.batch_size, cm.n_workers
        )
        for image_batch in tqdm(dataloader):
            out = model.generate(image_batch.to(device))
            caption_batch = processor.batch_decode(
                out, skip_special_tokens=True
            )
            captions.extend(caption_batch)

        df = pd.DataFrame({'file_name': image_ids, 'caption': captions})

        if cm.save_captions:
            save_captions(path_save, file_name, df)

        return df

    @staticmethod
    def compute_image_embeddings_open_clip(
        config: DictConfig,
        image_ids: List[str],
        image_paths: List[Path]
    ) -> Tuple[List[str], np.ndarray, torch.nn.Module]:
        """"""
        logger = get_logger()
        cm = config.model.open_clip
        model_name, pretrained = cm.model_name, cm.pretrained
        path_save = Path().resolve().joinpath(
            config.image_embeddings, model_name + cm.path_suffix_image
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        if check_if_embeddings_exist(path_save):
            logger.info(
                'Image embeddings are already computed with the same config! '
                'Reading and returning already computed embeddings.'
            )
            image_ids, embeddings = load_embeddings(path_save)

            return image_ids, embeddings, model

        data_loader = DataLoaderCollection.get_open_clip_image_dataloader(
            image_paths, image_preprocess, cm.batch_size, cm.n_workers
        )
        model.eval()
        context_length = model.context_length
        vocab_size = model.vocab_size
        n_params = np.sum([int(np.prod(p.shape)) for p in model.parameters()])
        logger.info(
            f'Computing {len(image_paths):,} embeddings from images with '
            f'OpenClip model: {model_name} {pretrained}. Model parameters: '
            f'{n_params:,}, Context length: {context_length}, Vocab size: '
            f'{vocab_size}'
        )
        embeddings = []

        for image_batch in tqdm(data_loader):

            with torch.no_grad():
                # embeddings will be normalized during the evaluation
                embeddings_batch = model.encode_image(
                    image_batch.to(device), normalize=False
                ).cpu().detach().numpy()

            embeddings.append(embeddings_batch)

        embeddings = np.concatenate(embeddings, axis=0)

        if cm.save_embeddings:
            save_embeddings(path_save, image_ids, embeddings)

        return image_ids, embeddings, model

    @staticmethod
    def compute_text_image_embeddings_open_clip(
        config: DictConfig,
        image_ids: List[str],
        captions: Optional[List[str]],
        image_paths: List[Path]
    ) -> Tuple[List[str], np.ndarray, torch.nn.Module]:
        """Computes image and if passed text embeddings with an OpenClip
        model. If both captions and image_paths are provided, the embeddings
        will be concatenated.
        Based on the definition of the scenarios, there are three cases:
        1. only image embeddings are computed from MultiModalVitEmbedImage
        2. both image and text embeddings are already computed
        2. both image and text embeddings should be computed

        Args:
            config:
            image_ids:
            captions:
            image_paths:

        Returns:

        """
        logger = get_logger()
        cm = config.model.open_clip
        model_name, pretrained = cm.model_name, cm.pretrained
        path_save_img = Path().resolve().joinpath(
            config.image_embeddings,
            model_name + cm.path_suffix_image
        )
        path_save_text = Path().resolve().joinpath(
            config.text_embeddings,
            model_name + cm.path_suffix_text
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        context_length = model.context_length
        vocab_size = model.vocab_size
        n_params = np.sum([int(np.prod(p.shape)) for p in model.parameters()])

        if (
                check_if_embeddings_exist(path_save_img)
                and not check_if_embeddings_exist(path_save_text)
        ):
            logger.info(
                'Image embeddings are already computed with the same '
                'config! Reading and returning already computed '
                'embeddings.'
            )
            image_ids, image_embeddings = load_embeddings(path_save_img)
            batch_size = cm.batch_size
            tokenizer = open_clip.tokenizer
            text_embeddings = []
            logger.info(
                f'Computing {len(image_paths):,} embeddings from captions '
                f'with OpenClip model: {model_name} {pretrained}. '
                f'Model parameters: {n_params:,}, '
                f'Context length: {context_length}, Vocab size: {vocab_size}'
            )
            for i in tqdm(range(0, len(captions), batch_size)):
                text_batch = captions[i: i + batch_size]
                token_batch = tokenizer.tokenize(text_batch)

                with torch.no_grad():
                    # embeddings will be normalized during the evaluation
                    text_emb_batch = model.encode_text(
                        token_batch.to(device), normalize=False
                    )
                    text_emb_batch = text_emb_batch.cpu().detach().numpy()

                text_embeddings.append(text_emb_batch)

            text_embeddings = np.concatenate(text_embeddings, axis=0)

            if cm.save_embeddings:
                save_embeddings(path_save_text, image_ids, text_embeddings)
        elif (
                check_if_embeddings_exist(path_save_img)
                and check_if_embeddings_exist(path_save_text)
        ):
            logger.info(
                'Text and Image embeddings are already computed with the same '
                'config! Reading and returning already computed '
                'embeddings.'
            )
            image_ids, image_embeddings = load_embeddings(path_save_img)
            _, text_embeddings = load_embeddings(path_save_text)

        else:
            data_loader = DataLoaderCollection.get_open_clip_caption_image_dataloader(
                captions, image_paths, image_preprocess, cm.batch_size,
                cm.n_workers
            )
            logger.info(
                f'Computing {len(image_paths):,} embeddings from captions and '
                f'images with OpenClip model: {model_name} {pretrained}. '
                f'Model parameters: {n_params:,}, '
                f'Context length: {context_length}, Vocab size: {vocab_size}'
            )
            text_embeddings, image_embeddings = [], []
            tokenizer = open_clip.tokenizer

            for text_batch, image_batch in tqdm(data_loader):
                token_batch = tokenizer.tokenize(text_batch)

                with torch.no_grad():
                    # embeddings will be normalized during the evaluation
                    text_emb_batch = model.encode_text(
                        token_batch.to(device), normalize=False
                    )
                    text_emb_batch = text_emb_batch.cpu().detach().numpy()
                    img_emb_batch = model.encode_image(
                        image_batch.to(device), normalize=False
                    )
                    img_emb_batch = img_emb_batch.cpu().detach().numpy()

                text_embeddings.append(text_emb_batch)
                image_embeddings.append(img_emb_batch)

            text_embeddings = np.concatenate(text_embeddings, axis=0)
            image_embeddings = np.concatenate(image_embeddings, axis=0)

            if cm.save_embeddings:
                save_embeddings(path_save_text, image_ids, text_embeddings)
                save_embeddings(path_save_img, image_ids, image_embeddings)

        embeddings = np.concatenate(
            (text_embeddings, image_embeddings), axis=1
        )
        return image_ids, embeddings, model

    @staticmethod
    def evaluate_embeddings(
        df: pd.DataFrame,
        image_ids: List[str],
        embeddings: np.ndarray,
        top_k: int
    ) -> Dict[str, float]:
        """"""
        logger = get_logger()
        logger.info(f'Evaluating embeddings with parameter top_k: {top_k - 1}')
        image_ids = np.array(image_ids)
        labels = df.loc[(df.file_name.isin(image_ids))].name.to_numpy()

        logger.info('Creating indices for semantic search...')
        embedding_dimension = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimension))
        faiss.normalize_L2(embeddings)
        index.add_with_ids(embeddings, np.arange(len(embeddings)))

        logger.info('Performing semantic search for the embedding corpus...')
        start = time.time()
        _, retrieved_ids = index.search(embeddings, k=top_k)
        logger.info(
            f'Semantic search took {round(time.time() - start, 2)} seconds...'
        )

        filtered_retrieved_ids = []

        for idx, row in enumerate(tqdm(retrieved_ids)):

            filtered_row = [x for x in row if x != idx]

            if len(filtered_row) != top_k - 1:
                filtered_row = filtered_row[:top_k - 1]

            filtered_retrieved_ids.append(filtered_row)

        filtered_retrieved_ids = np.array(filtered_retrieved_ids)

        matches = (
            np.expand_dims(labels, axis=1) == labels[filtered_retrieved_ids]
        )
        matches = torch.tensor(np.array(matches), dtype=torch.float16)
        targets = torch.ones(matches.shape)
        indexes = torch.arange(matches.shape[0]).view(
            -1, 1) * torch.ones(1, matches.shape[1]).long()

        metrics = [
            torchmetrics.RetrievalMRR(),
            torchmetrics.RetrievalNormalizedDCG(),
            torchmetrics.RetrievalPrecision(),
            torchmetrics.RetrievalRecall(),
            torchmetrics.RetrievalHitRate(),
            torchmetrics.RetrievalMAP()
        ]
        results = {}

        for metr in metrics:
            score = round(metr(targets, matches, indexes).item(), 4)
            metr_name = metr.__class__.__name__.replace('Retrieval', '')
            results[metr_name] = score
            logger.info(f'{metr_name}: {score}')

        return results
