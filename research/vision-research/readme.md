# vision-research

This project is a framework to run retrieval and ranking experiments on the [COCO](https://cocodataset.org/#download) 
and [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) datasets, 
by computing embedding vectors from the image and image caption modalities.

The framework is easily extendable to work on any dataset that satisfies the requirements (described later),
by implementing the necessary preprocessing steps for a given dataset and adding the corresponding 
preprocessing configuration file.
To set up the project, download the following data and place them under the **_/datasets/coco2017/_** folder.

- http://images.cocodataset.org/zips/train2017.zip
- http://images.cocodataset.org/zips/val2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

For the Open Images V7 dataset, see the notebook in the **_/notebooks_** folder. The Localized Narratives (captions), Human-verified labels 
and Class Names can be downloaded from [here](https://storage.googleapis.com/openimages/web/visualizer/index.html) and the images from
[here](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations).

## Experiments

Each experiment either computes embeddings from the modalities of the datasets
(images and captions) or loads the embeddings if they are already computed, 
detected by the same configured folder name. The embeddings are then evaluated 
by performing embedding retrieval with all the computed embedding vectors, and then computing
ranking metrics from the retrieved vectors. The vector search is [Faiss](https://github.com/facebookresearch/faiss)
based and the metrics are computed with [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/).

In the following, a mapping for (**scenario descriptions** → _experiment class names_) are defined.
The experiment runners can be found as **_ExperimentRunner_** classes in **_experiments.py_**:

1. **Make an LLM with vision capabilities describe an image, and embed the result** → _EmbedText_: embed COCO captions or generated captions with [Sentence Transformers](https://www.sbert.net/)
2. **Use an unimodal image neural network’s embeddings** → _EmbedImage_: embed COCO images with [timm](https://github.com/huggingface/pytorch-image-models)
3. **Additionally, use semantic embedding to encode textual metadata** → _EmbedTextImage_: combination of the previous two by vector concatenation
4. **Multimodal Vision transformer to encode the image** → _MultiModalVitEmbedImage_: embed COCO images with [OpenClip](https://github.com/mlfoundations/open_clip) models
5. **Multimodal Vision transformer to encode the image and metadata** → _MultiModalVitEmbedImageText_: embed both images and captions with OpenClip models
6. **ConcatSentenceTransformerOpenCLIP**: Embeds the images with an OpenCLIP model and the captions with a Sentence Transformer, and concatenates the results. 
- _CaptionImagesBlip_: generate image captions for COCO images with [Salesforce BLIP](https://huggingface.co/models?search=Salesforce/blip) models  (the generated captions can be embedded with experiments 1., 3., 5.)

The experiment results are logged with [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html),
based on the configured parameters.

## Configuration

The experiments are configuration driven with [Hydra](https://hydra.cc/docs/intro/).
The main settings in the configuration files are the following:

**config.yaml**

Select the appropriate preprocess config file in the defaults section.

```
defaults:
  - _self_
  - preprocess: coco_preprocess / open_images_v7_preprocess
  - model: model
  - pipeline: pipeline
  
mlflow:
    experiment_name: This should be a name of an ExperimentRunner class. 
                     It is used to identify which experiment to run and it 
                     is also identical with the MLflow Experiment that will 
                     store the results.
    run_name: mlflow run ID
    description: mlflow run description
    tags: mlflow tags as a list of tags
top_k: number of vectors to retrieve for evaluation
```

Paths to save / load embeddings and save experiment results. The paths to the embeddings and to the captions should be set for the relevant dataset
(See the structure from the computed artifacts, detailed later).

```
image_embeddings:
text_embeddings:
image_captions:
experiments:
```

**preprocess.yaml**

Options are explained in the file itself. There are options to remove duplicated labels
(when one object class is present multiple times on an image), to drop images where there is only one object class,
to combine the train and validation COCO datasets, and to replace the COCO captions with BLIP generated captions. 
For the Open Images V7 dataset, there is an option to remove the most frequent label sets that count as outliers.

**models.yaml**

Each model section can select a model for the previously defined experiments.
```
sentence_transformer:
pytorch_image_model:
image_captioning:
open_clip:
```

Each model configuration has the following parameters:

```
save_embeddings: true
```

to control if the computed embeddings will be saved or just passed to the evaluation step.

```
path_suffix:
```

Option to be able to store the computed embeddings in another path than the default, which is composed
of the

```
model_name:
```

Which identifies a given model to be downloaded and run from the leaderboards (see in the file).

**pipeline.yaml**

If this file is empty, the main configuration will be used from **config.yaml** to run an experiment. 
Otherwise, a list of experiments can be defined here, where each parameter in an experiment 
will replace the same parameter from the main config - the parameters that are not listed will stay
as they are in the main config. 

For example:

```
- mlflow:
    experiment_name: EmbedText
    run_name: multi-qa-mpnet-base-dot-v1
    description: COCO text embedding based retrieval based on concatenated caption
    tags:
        experiment: EmbedText
        dataset: COCO
        model: multi-qa-mpnet-base-dot-v1
        top_k: 10
  model:
    sentence_transformer:
        model_name: multi-qa-mpnet-base-dot-v1
        path_suffix: '_concat_captions'
```

The main config's *mlflow:* parameters and the *model:* config's *sentence_transformer:* parameters
will be updated to run an experiment.

## Installation

One option is using conda, but feel free to use pyenv, too.

`conda create -n vision_research python=3.10`

`conda activate vision_research`

`pip install -r requirements.txt`

## Start mlflow server based on the default config

`mlflow server --backend-store-uri artifacts/experiments/`

## Run the experiments based on the config

`python experiments.py`

## Share / load pre-computed artifacts

Download the already computed caption embeddings - "embeddings/text/", image embeddings -
"embeddings/image", image captions (computed with image captioning models) - 
"image_captions/", and experiment results - "experiments/" from
[this Google Cloud Storage link](https://console.cloud.google.com/storage/browser/superlinked-vectorhub-vision-research-embeddings),
and place them under the "./artifacts/" folder. Experiments can be run by utilizing the already
computed embeddings and captions and the experiments can be visualized from the MLflow artifacts.

## Evaluation logic background

The COCO and Open Images V7 datasets were chosen to satisfy the following two criteria.

1. (query, multiple answers pairs) - for each query vector, there should be at least one answer. 
2. Both the "query" and "multiple answers" should have <image, text metadata>.

The preprocessed datasets satisfy these conditions, because on each image there is at least one object class, 
and in the whole image corpus, there is at least one corresponding image that contains the same object class 
(e.g. man, motorcycle, road).

In each experiment, first the data is embedded - text / image / text + image, and the resulting
embedding vectors collectively form a vector space. When the ranking metrics are computed, there is an iteration
on this vector space, by selecting each vector as a query, and searching for the **_k (=10)_** most similar
vectors in the vector space. From the retrieved vectors, the ranking metrics are computed. A retrieval
hit is when for a given query the same object is retrieved - e.g. **_(man, motorcycle, road) → (man, motorcycle, road)._** 
