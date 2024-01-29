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


In this project, we explore the use of the COCO and Open Images V7 datasets to create embeddings from both images and text. 
We'll walk through various experiments, showcasing different embedding models and assessing their effectiveness 
with ranking metrics. This will give you a clear understanding of how multimodal data can be used in 
embedding processes.

## Dataset Specification

There are two essential criteria that a dataset must satisfy:

1. The dataset should be structured to have <query, multiple answers> pairs.
2. Both the "query" and "multiple answers" should include <image, text metadata>.

Finding publicly available datasets that meet these criteria is challenging. However, the 
[Common Objects in Context](https://cocodataset.org/#home) (COCO) and 
[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) datasets stand out as a notable exceptions. 
Both datasets are extensively utilized as benchmark datasets for object detection, segmentation, and image captioning tasks. 

COCO comprises images from 80 object categories, each accompanied by 5 unique, human-written captions. These captions 
distinctively describe the objects present in the images. Open Images V7 encompasses a significantly larger number of distinct 
object categories, totaling approximately 20,245. Diverging from mere captions, this dataset introduces Localized 
Narratives—a form of human audio description for each image segment, identified by mouse hovering. Each subpart of the 
Localized Narrative is accompanied by a timestamp. An illustrative example can be found 
[here](https://blog.research.google/2020/02/open-images-v6-now-featuring-localized.html). In this research, we 
leverage the textual representation of these Localized Narratives as captions.

These datasets fulfill our requirements by allowing a transformation where, for any given image and its set of objects 
(e.g., keyboard, mouse, person, TV), there exists at least one other image with an identical set of objects. This is achieved 
by identifying unique object sets and excluding those that appear only once. The Open Images V7 dataset is down-sampled 
by removing the outliers based on the frequency distribution of the label sets. Following these transformations, 
the COCO and the Open Images V7 datasets contain 103,429 and 149,847 samples, respectively.

The reference image for the mentioned object set from the COCO dataset is shown, and below, its corresponding 
human-written captions are displayed.

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

In each scenario, we vectorize/embed either the images or their captions, or both modalities. In cases 
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
- Using an AI model for generating captions, such as [Salesforce BLIP](https://arxiv.org/pdf/2201.12086.pdf), to compare 
AI-generated descriptions against human-written ones. BLIP models are accessible [here](https://huggingface.co/models?search=Salesforce/blip).

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_coco.png)

The table presents the Run Name, metrics values, number of model parameters, model name, and **_top k_** retrieved vectors 
for evaluation and a flag to indicate if the caption is generated or not. Utilizing all five human-written captions as a 
concatenation yielded the best results. The models "all-distilroberta-v1," "bge-large-en-v1.5", and "e5-large-v2" showed 
comparable performance in terms of MRR and NDCG metrics. Opting for a randomly chosen caption emerged as the second-best 
option on average, while generating new captions with BLIP resulted in the lowest MRR scores. Let's verify if these results 
hold true for the more diverse Open Images V7 dataset, which feature a broader range of objects and more descriptive 
Localized Narratives for each image. 

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_oiv7.png)

 BLIP-generated captions demonstrated greater efficiency on this dataset compared to human Localized Narratives. 
 Despite the previously mentioned models maintaining their relative order, the "all-mpnet-v2" model claimed the top 
 spot with an MRR score of 0.0662. Overall, the models performed similarly on the generated captions, with slight 
 variations that can be attributed to the Approximate Nearest Neighbor Search using FAISS.

In our experimentation with [LLaVA](https://arxiv.org/pdf/2304.08485.pdf), specifically version 1.5, for generating 
image captions, we noted that when requesting detailed descriptions for each image, the model tended to hallucinate 
non-existing objects, accounting for at least 50% of the generated text. Alternatively, when prompting the model 
to generate detailed descriptions only for elements it was confident about, it produced one or two short sentences, with 
no significant improvement over BLIP. Additionally, we explored GPT-4, which exhibited promise for all tested images. 
However, the current API rate limit renders it impractical for re-captioning datasets, estimating a duration of around 
2 weeks (as of the time of this writing)

Consistent performance is observed with Sentence Transformer models across diverse datasets. Generating captions 
with BLIP seems to be a viable option, especially when the captions provide a detailed description of each image. 
However, this fine granularity may reduce the retrieval capabilities of the system when such details are not essential, 
and the emphasis is on the overall concept.

### 2. Embedding Images with Larger Models

In this second scenario, we utilize [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) (timm) to embed 
each image, focusing solely on visual features for evaluation. We investigate how an increase in the number of model parameters 
impacts the quality of the embeddings and the subsequent evaluation results. The 
[ImageNet leaderboard](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-real.csv) within 
the timm repository serves as a resource for selecting models. Our comparison specifically looks at different sizes within the 
[EfficientNetV2](https://arxiv.org/pdf/2104.00298.pdf) family and includes a [Vision Transformer](https://arxiv.org/pdf/2010.11929v2.pdf)
(ViT) and its variants for contrast.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_image_coco.png)

Achieving the highest efficiency, the [caformer_m36](https://arxiv.org/pdf/2210.13452.pdf) model with approximately 
56 million parameters attained an MRR score of 0.368. The EfficientNetv2 family, 
specifically the smallest model with around 21.5 million parameters, proved to be the second most effective method.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_image_oiv7.png)

The smallest EfficientNetv2 model claimed the first position on the Open Images V7 dataset, with the caformer_m36 
model securing the second spot, followed by the EfficientNetv2 family models of sizes m and l.

The models' performance relative to each other remained consistent across datasets. Despite initial expectations of 
superior performance from the [Data-efficient Image Transformer models](https://arxiv.org/abs/2012.12877), attributed 
to their inductive biases from knowledge distillation, the experiments did not validate this assumption, resulting 
in these models occupying the last positions on both datasets.

### 3. Embedding Both Images and Their Captions

This approach integrates the embeddings from the first two scenarios using vector concatenation to form a combined embedding space. 
Here, we iterate through this space to retrieve the k nearest neighbors for each concatenated vector.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_image.png)

Combining vectors from **two unaligned vector spaces**  by concatenation did not yield performance improvements over the 
Computer Vision models. Specifically, in the case of COCO, the Sentence Transformers' performance deteriorated to the level 
of the Computer Vision models. The forthcoming multi-modal experiments will explore the impact of jointly training text and 
image encoders, with an expectation that concatenating their results may enhance performance compared to using the vectors 
from either of the encoders.

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


The performance of the tested models remained consistent across both datasets. ViT-based models outperformed the 
[ResNet50](https://arxiv.org/abs/1512.03385)-based model on COCO, while on the Open Images V7 dataset, the difference was not as 
significant, despite ViT models having more than 4 times as many parameters. Additionally, we present results from BLIP, which 
utilizes a ViT model for image encoding.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_blip.png)

BLIP achieved the best MRR scores on both datasets, surpassing the OpenCLIP models, aligning with the findings in the BLIP paper. 
The larger BLIP model has parameters totaling 447 million, while the base model has 224.7 million parameters. Notably, the largest 
BLIP model reached MRR scores of 0.494 and 0.112 on COCO and Open Images V7, respectively.

### 5. Embedding both images and their captions with Multimodal Vision Transformers

In this final scenario, we leverage both the Text and Image encoders from the CLIP and BLIP models to encode the captions 
and images separately, then concatenate these embeddings. A key difference from scenario 3. is that, here, the encoders 
have undergone joint pre-training in the context of CLIP or have been aligned with additional layers in the case of BLIP.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_text.png)

The order of the ViT-based OpenCLIP models is interchanged in these experiments, but their performance remains comparable. 
Overall, the individual model's performance is roughly consistent on both datasets. In the BLIP experiments, these models once 
again prove to be more efficient, with the largest model reaching an MRR score of 0.4953 and 0.112 on COCO and 
Open Images V7 datasets, respectively. 

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_text_blip.png)

In comparison to scenario 3, the concatenation of embeddings from two jointly trained or aligned encoders can boost 
retrieval performance. Upon observation of these datasets, the boost is more pronounced for OpenCLIP models.

### Comparison of All Results

Now, let's put all the results side by side for comparison.

![](assets/use_cases/retrieval_from_image_and_text/table_all_scenarios_coco.png)

![](assets/use_cases/retrieval_from_image_and_text/table_all_scenarios_oiv7.png)

In both the COCO and Open Images V7 datasets, the BLIP and OpenCLIP models (vision transformer models) emerged as the most efficient feature extractors. 
On the COCO dataset, the BLIP model achieved comparable performance when utilizing only the image modality. The top 
Sentence Transformers trailed by about 2% in MRR, but their inference speed is significantly faster. However, on the 
Open Images V7 dataset, the distinction was that the Sentence Transformers lagged behind the other models by around -37% in 
proportional MRR scores.

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

1. A closer look at [various image-captioning models](https://huggingface.co/models?other=image-captioning) to assess the quality 
of captions they generate, particularly in relation to the size of the models. How does the caption quality vary with the complexity 
of the models?
2. How well does GPT-4 perform in the task of image captioning?
3. What criteria should be employed to appropriately evaluate the effectiveness of modalities and determine if captions effectively 
convey image content for retrieval purposes?

## Conclusion

The experiments conducted in this project demonstrate that Transformer models are highly effective feature extractors for both textual 
and image data. State-of-the-art image-captioning models have proven to be excellent in annotating images and ensuring 
consistency across similar concepts. ConvNets, with their inherent inductive biases, emerge as robust 
feature encoders for image data. Moreover, the use of jointly trained text and image encoders appears to offer significant advantages 
in data embedding tasks involving multiple modalities, compared to the use of separately trained encoders and their combinations. 
Typically, BLIP and OpenCLIP models serve as reliable options for embedding data that involves both image and text modalities.

## Contributors

- [Kristóf Horváth, author](https://www.linkedin.com/in/kristof-horvath-0301/)