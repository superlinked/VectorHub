# Retrieval from Image and Text Modalities

## Problem statement

In the contemporary data-centric world, embeddings have become indispensable for converting complex and varied data into 
numerical representations that are both manageable and analytically powerful. Utilized across a spectrum of industries, 
from e-commerce to healthcare, these embeddings enable machines to interpret, analyze, and make predictions from 
large-scale datasets, encompassing both textual and visual information. Traditionally, models have relied on 
unimodal data, focusing on a single type of input. However, the advent of multimodal models, which synergize 
various data forms such as text and images, has proven to be a game-changer. These multimodal approaches 
surpass the limitations of unimodal methods, offering richer contextual insights and enhanced predictive capabilities. 
This shift towards multimodal data integration marks a significant stride in the field, paving the way for more 
sophisticated and accurate applications across diverse sectors. To explore how embeddings have started making 
a big impact across various industries, and to see some real-world examples of their growing popularity, 
check out [this link](https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/).


In this project, we explore the use of the COCO dataset to create embeddings from both images and text. 
We'll walk through various experiments, showcasing different embedding models and assessing their effectiveness 
with ranking metrics. This will give you a clear understanding of how multimodal data can be used in 
embedding processes.

## Dataset Specification

There are two essential criteria that a dataset must satisfy:

1. The dataset should be structured to have <query, multiple answers> pairs.
2. Both the "query" and "multiple answers" should include <image, text metadata>.

Finding publicly available datasets that meet these criteria is challenging. However, the 
[Common Objects in Context](https://cocodataset.org/#home) (COCO) dataset stands out as a notable exception. COCO is 
extensively utilized as a benchmark dataset for object detection, segmentation, and image captioning tasks. It comprises 
images from 80 object categories, each accompanied by 5 unique, human-written captions. These captions distinctively 
describe the objects present in the images.

The COCO dataset fulfills our requirements by allowing a transformation where, for any given image and its set of objects 
(e.g., keyboard, mouse, person, TV), there exists at least one other image with an identical set of objects. This is achieved 
by identifying unique object sets and excluding those that appear only once. The reference image for the mentioned object 
set is shown, and below, its corresponding human-written captions are displayed.

![](assets/use_cases/retrieval_from_image_and_text/reference_image.png)
[Reference image from the COCO dataset.](https://cocodataset.org/#home)

```
A young boy standing in front of a computer keyboard.
A little boy wearing headphones and looking at a computer monitor. 
He is listening intently to the computer at school.
A young boy stares up at the computer monitor.
A young kid with head phones on using a computer. 
```

## Scenarios

In each scenario, we vectorize/embed either the COCO images and their captions, or both modalities. In cases 
where both are used, the embeddings are concatenated. After embedding the entire dataset and normalizing each vector to 
unit length, we assess the quality of the embedding vectors through retrieval and by calculating ranking metrics.
This evaluation involves iterating over the vector space and, for each vector, retrieving its **_k (=10)_** nearest 
neighbors based on [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Cosine similarity, which 
measures the angle between two vectors, simplifies to a [dot product](https://en.wikipedia.org/wiki/Dot_product) 
calculation when the vectors are normalized, quantifying their alignment.

For the retrieved vectors, we calculate ranking metrics using [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html).
Our primary focus will be on [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) (MRR) and
[Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG). However,
you will see other information retrieval metrics like Mean Average Precision (MAP), Precision@k, and Recall@k appearing as well. 
MRR and NDCG are well explained [here](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg), while the
other metrics can be found [here](https://www.shaped.ai/blog/evaluating-recommendation-systems-part-1). No matter which metric is used, 
the effectiveness of the retrieval is always best when relevant items/hits are ranked at the top positions.

With this foundation, let's dive into the scenarios and their experimental results.

### 1. Embedding Image Captions

In this scenario, we employ the [Sentence-Transformers](https://www.sbert.net/) library to vectorize image captions. 
The [leaderboard](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/) is an excellent resource 
for selecting a top-performing model suited to our needs. Additionally, a wide range of models can be found on 
[Huggingface](https://huggingface.co/sentence-transformers). Beyond model selection, we explore different ways to process 
the textual data:

- Concatenating the 5 human-written image captions and embedding the combined text. Here, the Run Name is marked with a 
"_concat_captions" suffix.
- Randomly selecting one of the human-written image captions.
- Using an AI model for generating captions, such as [Salesforce BLIP](https://arxiv.org/pdf/2301.12597.pdf), to compare 
AI-generated descriptions against human-written ones. BLIP models are accessible [here](https://huggingface.co/models?search=Salesforce/blip).

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text.png)

The table presents the Run Name, metrics values, number of model parameters, model name, and **_top k_** retrieved vectors for evaluation. 
The least effective result was observed when captions were chosen randomly. The second-best performance was achieved by 
concatenating human-written captions. Surprisingly, the top-performing model was "all-mpnet-base-v2," which embedded captions 
generated by the "blip-image-captioning-base" model.

BLIP generated this caption for the reference image:

```
A young boy wearing headphones while using a computer.
```

Following the embedding of these generated captions, the images retrieved for the aforementioned query are:

![](assets/use_cases/retrieval_from_image_and_text/retrieved_images.png)
[Images retrieved from the COCO dataset in response to using the reference image as a query.](https://cocodataset.org/#home)

Based on the comparison between human-written and AI-generated captions, I hypothesize that the AI-generated captions are more 
efficient. This efficiency likely stems from their consistency across images sharing the same object set. AI models, such as 
the one used here, tend to generate descriptions that are more standardized and uniform for similar images, potentially leading 
to more effective embeddings in scenarios like ours.

### 2. Embedding Images with Larger Models

In this second scenario, we utilize [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) (timm) to embed 
each image, focusing solely on visual features for evaluation. We investigate how an increase in the number of model parameters 
impacts the quality of the embeddings and the subsequent evaluation results. The 
[ImageNet leaderboard](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-real.csv) within 
the timm repository serves as a resource for selecting models. Our comparison specifically looks at different sizes within the 
[EfficientNetV2 family](https://arxiv.org/pdf/2104.00298.pdf) and includes a [Vision Transformer](https://arxiv.org/pdf/2010.11929v2.pdf)
(ViT) for contrast.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_image.png)

The results table reveals that the ViT trails significantly behind the others, with the smallest EfficientNet 
(~21.5M parameters) producing the most effective embeddings. Interestingly, there's no straightforward correlation between model 
size and evaluation metrics. The second-best performer is the largest EfficientNet model (~118.5M parameters), outperforming 
the third model which has around 54M parameters. This variation in performance between transformer and Convolutional Neural Network 
(ConvNet) architectures can be attributed to the 
[strong inductive biases inherent in convolutional networks](https://ai.meta.com/blog/computer-vision-combining-transformers-and-convolutional-neural-networks/).

### 3. Embedding Both Images and Their Captions

This approach integrates the embeddings from the first two scenarios using vector concatenation to form a combined embedding space. 
Here, we iterate through this space to retrieve the k nearest neighbors for each concatenated vector.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_image.png)

Analyzing the results, it's evident that the combination of the smallest EfficientNet with the "all-mpnet-base-v2" model remains 
the top performer, similar to the pattern observed in scenario 2. This combination shows a notable improvement, with around 
a 9% and 10% increase in MRR and NDCG scores, respectively, when compared to the second-best run. These findings underscore the 
consistent effectiveness of a small model size across both individual and combined embedding scenarios.

### 4. Embedding Images with Multimodal Vision Transformers

In this scenario, we delve into models based on [Contrastive Language-Image Pretraining](https://arxiv.org/pdf/2103.00020.pdf) (CLIP). 
CLIP models are unique for their multimodal approach, featuring separate but jointly trained Text and Image encoders. This joint 
training fosters a multimodal space where semantically similar concepts, regardless of being text or image, are positioned closer 
together in the embedding space.

![](assets/use_cases/retrieval_from_image_and_text/clip.png)
[CLIP's high level architecture from the paper Learning Transferable Visual Models From Natural Language Supervision.](https://arxiv.org/pdf/2103.00020.pdf)

The structure depicted in the image illustrates that the encoders are versatile and adaptable to various model architectures for 
embedding text or image data. For our experiment, we utilize pretrained models from the 
[OpenClip leaderboard](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv) and apply the Image Encoder 
to embed the images, followed by an evaluation of the outcomes.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image.png)


In this experiment, the larger ViT Image Encoder, with over four times the parameters, outperformed the 
[ResNet50](https://arxiv.org/abs/1512.03385) Image Encoder, achieving about a 4% higher score in both MRR and NDCG. Notably, 
this is an 11% improvement compared to scenario 2., where vision models alone were used. Interestingly, while in scenario 2. the 
ConvNets significantly outperformed the ViT, this trend is reversed here, highlighting the potential 
advantages of larger, more complex ViT models in certain contexts.

### 5. Embedding both images and their captions with Multimodal Vision Transformers

In this final scenario, we leverage both the Text and Image encoders from the CLIP based models to encode the captions and images 
separately, then concatenate these embeddings. The key distinction from scenario 3. is that here, the encoders have been jointly pre-trained.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_text.png)

When compared to scenario 4., both runs in this scenario show an average improvement of 4% in both MRR and NDCG scores. This outcome 
clearly indicates the advantage of utilizing the information present in both captions and images, demonstrating the strength of jointly 
trained multimodal encoders in enhancing the quality of embeddings.

### Comparison of All Results

Now, let's put all the results side by side for comparison.

![](assets/use_cases/retrieval_from_image_and_text/table_all_scenarios.png)

The largest CLIP-based model, boasting around 427.6M parameters, reached MRR and NDCG scores of 0.494 and 0.532, respectively. 
An MRR score close to 0.5 implies that, on average, we can retrieve a matching object set in the 2nd position out of k = 10 options 
from the embedding space for a given query. However, it's noteworthy that the best result from scenario 1. is only 3% lower in terms 
of its MRR. This emphasizes yet again the effectiveness of the AI-generated captions.

We should also take into account the inference time and GPU demands for each scenario. These metrics were gathered using 
an [RTX 3080 16 GB GPU](https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621), capable of 29.77 TFLOPS on FP32. 
When processing the merged COCO training and validation dataset, containing 103,429 data samples post-preprocessing, we noted the 
following inference times and resource allocations. It's important to note that GPU utilization was always maximized through 
parallelized data loading to ensure efficiency.

- Embedding captions with "all-mpnet-base-v2" takes approximately 5 minutes and uses about 2.8 GB of GPU memory for batches of 128.
- Generating captions with "Salesforce/blip-image-captioning-base" spans around 3 hours and requires close to 15.5 GB of GPU memory, 
also with batches of 128.
- Embedding images with "tf_efficientnetv2_s.in21k_ft_in1k" similarly takes about 3 hours and consumes 15 GB of GPU memory for batch 
sizes of 128.
- Embedding both captions and images using the OpenCLIP "ViT-L-14_datacomp_xl_s13b_b90k" model can be completed in about 50 minutes when 
processing with a batch size of 512, requiring 14.5 GB of GPU memory.

Indeed, if high-quality image captions are already in hand, employing Sentence Transformers for embedding proves to be highly efficient. 
This method offers a balance of speed and effectiveness. On the other hand, if only images are available and captions need to be 
generated, the process can be time-consuming, which could be a significant factor when deciding on the method for a given application 
or project.

## Open Questions

The outcomes of these experiments open up several intriguing questions for further investigation. Here are a few key areas to explore:

1. The potential of other transformer-based architectures in improving results. For instance, [Data-Efficient Image Transformers](https://arxiv.org/pdf/2012.12877.pdf), 
utilize knowledge distillation and inductive biases from ConvNets for training. Would such models offer superior performance?
2. A closer look at [various image-captioning models](https://huggingface.co/models?other=image-captioning) to assess the quality 
of captions they generate, particularly in relation to the size of the models. How does the caption quality vary with the complexity 
of the models?
3. An examination of the consistency of ConvNets' performance across diverse datasets. Do these networks consistently perform well 
across various data environments, or does their effectiveness vary depending on the specific dataset?

## Conclusion

The experiments conducted in this project demonstrate that Transformer models are highly effective feature extractors for textual 
data. State-of-the-art image-captioning models have proven to be excellent in annotating images and ensuring 
consistency across similar concepts. ConvNets, with their inherent inductive biases, emerge as robust 
feature encoders for image data. Moreover, the use of jointly trained text and image encoders appears to offer significant advantages 
in data embedding tasks involving multiple modalities, compared to the use of separately trained encoders and their combinations. 

## Contributors

- [Kristóf Horváth, author](https://www.linkedin.com/in/kristof-horvath-0301/)