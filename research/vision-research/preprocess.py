from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Tuple, Union

from hydra import compose, initialize
from omegaconf.dictconfig import DictConfig
import pandas as pd
from PIL import Image
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from transformers.models.blip.processing_blip import BlipProcessor

from utils import read_json, get_logger


class PreProcessor(ABC):
    @staticmethod
    @abstractmethod
    def preprocess(config: DictConfig) -> pd.DataFrame:
        pass


class COCOPreProcessor(PreProcessor):

    @staticmethod
    def create_coco_dataset_df(
        config: DictConfig,
        split: Literal['train', 'valid']
    ) -> pd.DataFrame:

        logger = get_logger()
        cp = config.preprocess
        path_data = Path().resolve().joinpath(*list(cp.path_data))
        path_annotations = path_data.joinpath(*list(cp.dir_annotations))
        logger.info(f"Creating DataFrame from the COCO {split} dataset...")

        if split == 'train':
            path_images = path_data.joinpath(*list(cp.dir_train_images))
            path_captions = path_annotations.joinpath(cp.captions_train)
            path_instances = path_annotations.joinpath(cp.instances_train)
        else:
            path_images = path_data.joinpath(*list(cp.dir_valid_images))
            path_captions = path_annotations.joinpath(cp.captions_valid)
            path_instances = path_annotations.joinpath(cp.instances_valid)

        logger.info("Reading data of instances and captions...")
        coco_captions = read_json(path_captions)
        coco_instances = read_json(path_instances)

        df_coco_img = pd.DataFrame(coco_instances['images'])
        df_coco_ann = pd.DataFrame(coco_instances['annotations'])
        df_coco_cats = pd.DataFrame(coco_instances['categories'])
        df_coco_caps = pd.DataFrame(coco_captions['annotations'])

        df_coco_caps.drop(columns=['id'], inplace=True)
        df_coco_img.rename(columns={'id': 'image_id'}, inplace=True)
        df_coco_cats.rename(columns={'id': 'category_id'}, inplace=True)

        logger.info("Merging category IDs with category names...")

        df_coco_imgs_cats = pd.merge(
            df_coco_ann[['image_id', 'category_id']],
            df_coco_cats,
            how='left',
            on='category_id'
        )
        if cp.remove_duplicated_labels:
            df_coco_imgs_cats.drop_duplicates(
                subset=['image_id', 'name'], inplace=True
            )

        df_coco_imgs_cats = df_coco_imgs_cats.groupby('image_id')['name'].agg(
            lambda x: ','.join(sorted(list(x)))
        ).reset_index()

        logger.info("Merging image data with categories...")
        df = pd.merge(
            df_coco_img[['file_name', 'height', 'width', 'image_id']],
            df_coco_imgs_cats,
            how='left',
            on='image_id'
        )
        logger.info("Merging previous results with captions...")

        if cp.concatenate_captions:
            logger.info("Concatenating COCO captions for each image...")
            df_coco_caps = df_coco_caps.groupby('image_id')['caption'].agg(
                lambda x: ' '.join(x)
            ).reset_index()
        else:
            logger.info("Choosing one caption for each image randomly...")
            df_coco_caps = df_coco_caps.groupby('image_id').sample(n=1)

        df = pd.merge(df, df_coco_caps, how='left', on='image_id')
        df['image_path'] = df.file_name.apply(lambda x: path_images.joinpath(x))

        return df

    @staticmethod
    def preprocess(config: DictConfig) -> pd.DataFrame:

        logger = get_logger()
        cp = config.preprocess
        df = COCOPreProcessor.create_coco_dataset_df(config, 'train')
        logger.info(f'Train samples: {len(df):,}')

        if cp.combine_train_val:
            df_vl = COCOPreProcessor. create_coco_dataset_df(config, 'valid')
            logger.info(f'Validation samples: {len(df_vl):,}')
            df = pd.concat([df, df_vl], axis=0)

        # select labels which have a support of at least 2
        cats_counts = df.name.value_counts()
        classes_with_support = cats_counts[cats_counts > 1].index.tolist()

        df = df.loc[(df.name.isin(classes_with_support))].reset_index(drop=True)
        logger.info(
            f'{len(classes_with_support)} overall combined classes and '
            f'{len(df):,} samples to be vectorized!'
        )
        if cp.drop_samples_with_one_class:
            logger.info('Removing images with only one class...')
            df['n_classes'] = df.name.apply(lambda x: x.count(','))
            df = df.loc[df.n_classes > 1].reset_index(drop=True)
            df.drop(columns=['n_classes'], inplace=True)

        if cp.replace_captions:
            logger.info(
                f"Replacing COCO captions with generated {cp.generated_captions}!"
            )
            df.drop(columns='caption', inplace=True)
            path_captions = Path().resolve().joinpath(config.image_captions)
            df_captions = pd.read_csv(
                path_captions.joinpath(cp.generated_captions)
            )
            df = pd.merge(df, df_captions, how='inner', on='file_name')

        return df


class OpenImagesV7PreProcessor(PreProcessor):
    @staticmethod
    def preprocess(config: DictConfig) -> pd.DataFrame:
        logger = get_logger()
        cp = config.preprocess
        path_data = Path().resolve().joinpath(*list(cp.path_data))
        path_images = path_data.joinpath(cp.dir_images)
        path_df = path_data.joinpath(cp.dataframe)

        df = pd.read_parquet(path_df)
        df['image_path'] = df.file_name.apply(lambda x: path_images.joinpath(x))

        logger.info(
            f"{len(df):,} Open Images V7 samples...\n"
            f"Removing samples where label set support is less than 2..."
        )
        # remove samples where label set support < 2
        df_freq = df.name.value_counts().reset_index(name='freq')
        classes_with_support = set(
            df_freq.loc[(df_freq.freq >= 2)].name.tolist()
        )
        df = df.loc[(df.name.isin(classes_with_support))].reset_index(drop=True)
        logger.info(f"{len(df):,} remaining samples...\n")

        if cp.remove_outliers:
            logger.info("Removing outliers based on label set frequencies...")
            df_freq = df.name.value_counts().reset_index(name='freq')
            mean, std = df_freq.freq.mean(), df_freq.freq.std()

            df_filtered_freq = df_freq.loc[
                (df_freq.freq > mean - 3 * std)
                & (df_freq.freq < mean + 3 * std)
            ].reset_index(drop=True)

            classes_with_support = set(df_filtered_freq.name.tolist())
            df = df.loc[
                (df.name.isin(classes_with_support))
            ].reset_index(drop=True)

            logger.info(
                f"{len(df):,} remaining samples after outlier removal..."
            )

        if cp.replace_captions:
            logger.info(
                f"Replacing Open Images V7 localized narratives with "
                f"generated {cp.generated_captions}!"
            )
            df.drop(columns='caption', inplace=True)
            path_captions = Path().resolve().joinpath(config.image_captions)
            df_captions = pd.read_csv(
                path_captions.joinpath(cp.generated_captions)
            )
            df = pd.merge(df, df_captions, how='inner', on='file_name')

        logger.info(f"{df.name.nunique():,} unique labels...")

        return df


def factory(
    config: DictConfig
) -> Union[COCOPreProcessor, OpenImagesV7PreProcessor]:

    return globals()[config.preprocess.preprocessor]


class DataLoaderCollection:
    @staticmethod
    def get_timm_image_dataloader(
        image_paths: List[Path],
        model: torch.nn.Module,
        batch_size: int,
        n_workers: int
    ) -> DataLoader:

        image_size = model.default_cfg['input_size'][1]
        mean, std = model.default_cfg['mean'], model.default_cfg['std']

        dataset = TimmImageDataset(image_paths, image_size, mean, std)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=n_workers
        )
        return data_loader

    @staticmethod
    def get_blip_image_dataloader(
        image_paths: List[Path],
        processor: BlipProcessor,
        batch_size: int,
        n_workers: int
    ) -> DataLoader:

        dataset = BlipImageDataset(image_paths, processor)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=n_workers
        )
        return data_loader

    @staticmethod
    def get_open_clip_image_dataloader(
        image_paths: List[Path],
        transform: transforms.Compose,
        batch_size: int,
        n_workers: int
    ) -> DataLoader:

        dataset = OpenClipImageDataset(image_paths, transform)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=n_workers
        )
        return data_loader

    @staticmethod
    def get_open_clip_caption_image_dataloader(
        captions: List[str],
        image_paths: List[Path],
        transform: transforms.Compose,
        batch_size: int,
        n_workers: int
    ) -> DataLoader:

        dataset = OpenClipCaptionImageDataset(
            captions, image_paths, transform
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=n_workers
        )
        return data_loader


class TimmImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        image_size: int,
        mean: Tuple[int, int, int],
        std: Tuple[int, int, int]
    ):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(size=(image_size, image_size)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            image = read_image(str(image_path), mode=ImageReadMode.RGB)
        except RuntimeError:
            # RGBA image
            image = read_image(str(image_path))[:3, ...]

        image = image.to(torch.float32)
        image = self.transform(image)

        return image


class BlipImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        processor: BlipProcessor
    ):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.processor(image, return_tensors='pt')['pixel_values'][0]

        return image


class OpenClipImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image


class OpenClipCaptionImageDataset(Dataset):
    def __init__(
        self,
        captions: List[str],
        image_paths: List[Path],
        transform: transforms.Compose
    ):
        self.captions = captions
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return caption, image


if __name__ == '__main__':

    initialize(config_path='configs')
    config = compose(config_name='config')
    df = factory(config).preprocess(config)
    print(df.isnull().sum())
    print(df.head())
