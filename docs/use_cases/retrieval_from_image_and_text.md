# Retrieval from Image and Text Modalities

## The value of multimodal embedding

In our contemporary data-centric world, embeddings have become indispensable for converting complex and varied data into numerical representations that are both manageable and analytically powerful. [Across a spectrum of industries](https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/), from e-commerce to healthcare, these embeddings enable machines to interpret, analyze, and make predictions from large-scale datasets containing textual and/or visual information. Traditionally, models have relied on unimodal data, typically either images or text, but not both. However, the advent of multimodal models, which can synergize various data forms, has proven to be a game-changer. **Multimodal approaches surpass the limitations of unimodal methods, offering richer contextual insights and enhanced predictive capabilities, and paving the way for more sophisticated and accurate applications across diverse sectors**.

Below, we carry out various text and image embedding experiments using COCO and Open Images V7 datasets, showcasing different unimodal and multimodal embedding models, and assessing their effectiveness using ranking metrics. By the end, you'll have an understanding of how to embed multimodal data. We'll also evaluate the performance of unimodal vs. multimodal embeddings, and how different multimodal models stack up against each other.

## Our datasets: COCO and Open Images V7

Our dataset must satisfy two essential criteria:

1. The dataset should be structured to have <query, multiple answers> pairs.
2. Both the "query" and "multiple answers" should include <image, text metadata>.

Publicly available datasets that meet these criteria are rare. [Common Objects in Context](https://cocodataset.org/#home) (COCO) and [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) are notable exceptions. Both datasets are extensively utilized as benchmark datasets for object detection, segmentation, and image captioning tasks. 

COCO comprises images from 80 object categories, each image accompanied by 5 unique, human-written captions that distinctively describe objects present in the image. Open Images V7 encompasses a significantly larger number of distinct object categories - approximately 20,245. In addition to captions, Open Images V7 introduces Localized Narratives - human audio descriptions - for each image segment, identified by mouse hovering. Each subpart of the Localized Narrative is accompanied by a timestamp. An illustrative example can be found [here](https://blog.research.google/2020/02/open-images-v6-now-featuring-localized.html). In our experiments, we leverage the textual representation of these Localized Narratives as captions.

COCO and Open Images V7 fulfill our essential dataset criteria; we can identify which images contain object sets (e.g., keyboard, mouse, person, TV) in any particular image, and ensure that at least two images have the identical object set by excluding images with object sets that appear only once. Based on label set frequency distribution, these outliers are removed from both Open Images V7 and COCO datasets. The resulting down-sampled COCO and the Open Images V7 datasets contain 103,429 and 149,847 samples, respectively.

Here's an example image from the COCO dataset, and below it, the human-written captions corresponding to the image's object set.

![COCO dataset example image](assets/use_cases/retrieval_from_image_and_text/reference_image_COCO.png)
*Example image from the [COCO dataset](https://cocodataset.org/#home).*

```
A young boy standing in front of a computer keyboard.
A little boy wearing headphones and looking at a computer monitor. 
He is listening intently to the computer at school.
A young boy stares up at the computer monitor.
A young kid with head phones on using a computer. 
```

## Our embedding experiments

In our experiments below, we **vectorize/embed**, respectively, 1) image captions, 2) images, 3) both images and their captions, 4) images with multimodal transformers, 5) both images and their captions with multimodal transformers. In cases where images and their captions are vectorized separately, the embeddings are concatenated.

After embedding the entire dataset and normalizing each vector to unit length, we **assess the quality of the embedding vectors by retrieving them and calculating ranking metrics**. More specifically. we iterate over the vector space and retrieve each vector's **_k (=10)_** nearest neighbors based on [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Cosine similarity measures and quantifies the angle between two vectors by simplifying to a [dot product](https://en.wikipedia.org/wiki/Dot_product) calculation.

For the retrieved vectors, we calculate ranking metrics using [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html). We focus primarily on [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) (MRR) and [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG), both of which you can read more about [here](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg). But we also use other information retrieval metrics like Mean Average Precision (MAP), Precision@k, and Recall@k, which are explained in detail [here](https://www.shaped.ai/blog/evaluating-recommendation-systems-part-1). In all of these metrics, the higher the ranking of relevant items/hits, the more effective the retrieval.

Now that we understand the basics, let's dive into each of our embedding experiments and their results. Afterwards, we'll put these results side by side to compare them.

### 1. Embedding image captions

In experiment 1, we vectorized image captions using the Sentence-Transformers library, selecting top-performing models suited to our use case from [SBERT Pretrained Models Leaderboard](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/) as well as [Huggingface Sentence Transformers](https://huggingface.co/sentence-transformers). In addition to using different models, we tried different ways of processing our textual data in different types of runs:

- Concatenating the 5 human-written image captions and embedding the combined text. All these runs are marked with a "_concat_captions" suffix in the table below.
- Randomly selecting one of the human-written image captions. All these runs are marked with "_random_caption."
- Using an AI model to generate captions - we selected [Salesforce BLIP](https://arxiv.org/pdf/2201.12086.pdf), for comparison against human-written ones. You can find BLIP models [here](https://huggingface.co/models?search=Salesforce/blip).

We collected the results of experiment 1 in the table below, which presents the Run Name, metrics values, number of model parameters, model name, and **_top k_** retrieved vectors for evaluation, as well as a flag to indicate whether the caption is generated or not.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_coco.png)

Concatenating the 5 human-written captions yielded the best results. The "all-distilroberta-v1," "bge-large-en-v1.5", and "e5-large-v2" models performed comparably well on MRR and NDCG metrics. Using a randomly chosen caption produced the second best outcomes on average. Generating new captions with BLIP produced the lowest MRR scores.

But do these outcome patterns hold true for the more diverse Open Images V7 dataset, which features a broader range of objects and more descriptive Localized Narratives for each image? Let's take a look in the table below.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_oiv7.png)

BLIP-generated captions were more efficient than human Localized Narratives on the Open Images V7 dataset. The "all-distilroberta-v1," "bge-large-en-v1.5", and "e5-large-v2" models maintained their relative performance order, but the "all-mpnet-v2" model did better, with the top MRR score: 0.0706. Overall, all the models performed comparably on generated captions, with slight variations attributable to Approximate Nearest Neighbor Search using [FAISS](https://github.com/facebookresearch/faiss).

When we used [LLaVA](https://arxiv.org/pdf/2304.08485.pdf) 1.5 to generate detailed descriptions for each image, the model tended to hallucinate non-existing objects at least 50% of the time. Performance improved when prompting for detailed descriptions of only those elements LLaVA 1.5 was confident about, but the model's one to two short sentence output was no better than BLIP's output. We also looked at GPT-4, which performed well for all tested images. But GPT-4's current API limit means that it would take an estimated 2 weeks to re-caption an entire dataset, making it impractical.

In sum, the **Sentence Transformers models performed consistently across diverse datasets in our first experiment**. In addition, **generating captions with BLIP seems to be a viable option, especially when the captions provide a detailed description of each image**. However, in use cases requiring descriptions that focus on the overall concept, and such fine granularity isn't necessary, BLIP-generated captions may unnecessarily reduce the system's retrieval capabilities.

### 2. Embedding images with larger models

In our second experiment, we used [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) (timm) to embed each image, and evaluated image embedding exclusively, looking at how an increase in the number of model parameters impacts the quality of the embeddings and subsequent performance. We selected our models from within the timm repository of [ImageNet leaderboard](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-real.csv). We compared different sizes within the [EfficientNetV2](https://arxiv.org/pdf/2104.00298.pdf) family, and included a [Vision Transformer](https://arxiv.org/pdf/2010.11929v2.pdf) (ViT) and its variants for contrast. First, let's look at **notable COCO dataset results**.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_image_coco.png)

On the COCO dataset, the [caformer_m36](https://arxiv.org/pdf/2210.13452.pdf) model, which has approximately 56 million parameters, achieved the highest efficiency with an MRR score of 0.368. The next most efficient models were the EfficientNetv2 family. Its smallest model, with around 21.5 million parameters, had the second highest MMR score, at 0.352. Now, let's see how the models performed **on the Open Images 7 dataset**.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_image_oiv7.png)

The smallest EfficientNetv2 model was the most efficient performer on the Open Images V7 dataset, caformer_m36 came second, followed by the EfficientNetv2 model, sizes m and l. The models' performance relative to each other remained roughly consistent across datasets. Also, though we expected superior performance from the [Data-efficient Image Transformer models (DeiTs)](https://arxiv.org/abs/2012.12877) because of their inductive biases (acquired through knowledge distillation), of all the models we tested on both datasets, DeiTs performed the most poorly.

### 3. Embedding both images and their captions

Our third experiment **concatenated vectors from our first two experiments into a combined vector space**. We iterated through this space to retrieve the k nearest neighbors for each concatenated vector, with the following results.

![](assets/use_cases/retrieval_from_image_and_text/table_embed_text_image.png)

Concatenating vectors from two unaligned vector spaces into one space - using the Sentence Transformers models on the COCO dataset, **deteriorated performance to the level of the Computer Vision models**. As a result, we next investigated (in experiments 4. and 5.) **whether using _jointly trained_ text and image encoders, and then concatenating their vectors, might lead to better performance than concatenating vectors created by _separately trained_ image and text encoders**.

### 4. Embedding images with Multimodal Transformers

In experiment 4, we look at the performance of models based on [Contrastive Language-Image Pretraining](https://arxiv.org/pdf/2103.00020.pdf) (CLIP). CLIP models employ separate but jointly trained Text and Image encoders to create a single multimodal embedding space. Regardless of whether the embeddings in this space represent text or image, if they are semantically similar, they are positioned closer together.

![](assets/use_cases/retrieval_from_image_and_text/clip.png)
*CLIP's high level architecture, from ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/pdf/2103.00020.pdf)*

The structure of CLIP encoders (image above) makes them versatile and adaptable to various model architectures for embedding text or image data. In our experiment, we used pretrained models from the [OpenClip leaderboard](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv), and applied the Image Encoder to embed the images. Then we evaluated the outcomes.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image.png)

The performance of the tested models was consistent across both datasets. ViT-based models outperformed the [ResNet50](https://arxiv.org/abs/1512.03385)-based model on COCO. On the Open Images V7 dataset, the difference between ViT and ResNet50 (RN50_openai) was less significant, despite ViT models having more than 4 times as many parameters. We also present results (below) from BLIP, which encodes images using a ViT model.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_blip.png)

**BLIP achieved the best MRR scores on both datasets**, surpassing the OpenCLIP models, aligning with findings of the [BLIP paper](https://arxiv.org/pdf/2201.12086.pdf). The larger of the two BLIP models, with 447 million parameters (the base model has 224.7 million), reached notable MRR scores of 0.494 on COCO and 0.112 on Open Images V7.

### 5. Embedding both images and their captions with Multimodal Transformers

In our final experiment, **we used Text and Image encoders from both CLIP and BLIP models to encode captions and images separately, then concatenated the resulting embeddings**. A key difference from our third experiment (embedding both images and their captions) is that, here, the **encoders have either undergone joint pre-training** - in the case of CLIP, **or been aligned with additional layers** - in the case of BLIP.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_text.png)

In experiment 5, the rank order of the two ViT-based OpenCLIP models on the COCO dataset was inverted (from what it was in experiment 4), but they performed comparably well - on both the COCO and Open Images V7 datasets. In the BLIP experiments (below), the BLIP models once again proved to be more efficient; the largest model had an MRR score of 0.4953 on the COCO dataset - marginally (0.26%) better than the best OpenCLIP model, and 0.112 on Open Images V7 - 7.07% better than the best OpenCLIP model.

![](assets/use_cases/retrieval_from_image_and_text/table_multimodal_vit_embed_image_text_blip.png)

Here, as we anticipated, **concatenating embeddings from two _jointly trained or aligned encoders_ boosted retrieval performance, over and above the results achieved by concatenating vectors created by _separately trained_ image and text encoders** (in experiment 4). This boost was more pronounced for the OpenCLIP models.

### Comparing all results

Now, let's put all our results side by side for comparison.

![](assets/use_cases/retrieval_from_image_and_text/table_all_scenarios_coco.png)

![](assets/use_cases/retrieval_from_image_and_text/table_all_scenarios_oiv7.png)

In both the COCO and Open Images V7 datasets, the BLIP and OpenCLIP models proved to be the most efficient feature extractors. On the COCO dataset, the BLIP model performed about the same using only image embeddings as it did when using both image and caption embeddings. Indeed, in general, using both image and caption embeddings makes the highest performing models perform only marginally better - regardless of whether the model embeds images or text. The top Sentence Transformers models' MRR scores trailed by about 2%, but their inference speed was significantly faster. However, on the Open Images V7 dataset, Sentence Transformers models' proportional MRR scores lagged behind the other models by around -37%.

We should also **take into account the inference time and GPU demands** for each of our experiments. These metrics were gathered using an [RTX 3080 16 GB GPU](https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621), capable of 29.77 TFLOPS on FP32. When processing the merged COCO training and validation dataset, containing 103,429 data samples post-preprocessing, we noted the following **inference times and resource allocations**. It's important to note that GPU utilization was always maximized through parallelized data loading to ensure efficiency.

- Embedding captions with "all-mpnet-base-v2" takes approximately 5 minutes and uses about 2.8 GB of GPU memory for batches of 128.
- Generating captions with "Salesforce/blip-image-captioning-base" spans around 3 hours and requires close to 15.5 GB of GPU memory, also with batches of 128.
- Embedding images with "tf_efficientnetv2_s.in21k_ft_in1k" similarly takes about 3 hours and consumes 15 GB of GPU memory for batch sizes of 128.
- Embedding both captions and images using the OpenCLIP "ViT-L-14_datacomp_xl_s13b_b90k" model can be completed in about 50 minutes when processing with a batch size of 512, requiring 14.5 GB of GPU memory.

If high-quality image captions are already in hand, embedding with Sentence Transformers proves to be highly efficient, and balances speed and effectiveness. On the other hand, if only images are available and your application or project also requires captions to be generated, the time cost of different methods should be considered carefully.

## Questions for further investigation

The outcomes of these experiments open up several intriguing questions for further investigation. Here are a few key areas to explore:

1. A closer look at [various image-captioning models](https://huggingface.co/models?other=image-captioning) to assess the quality of captions they generate, particularly in relation to the size of the models. How does the caption quality vary with the complexity of the model?
2. How well does GPT-4 perform at captioning images?
3. What criteria should be employed to evaluate the effectiveness of modalities and determine whether captions effectively convey image content for retrieval purposes?

## Conclusion

Our experiments demonstrate that Transformer models are highly effective feature extractors for both text and image data. State-of-the-art image-captioning models have proven to be excellent in annotating images and ensuring consistency across similar concepts. Vision Transformers emerge as robust feature encoders for image data. Moreover, using jointly trained text and image encoders appears to offer significant advantages in data embedding tasks involving multiple modalities, compared to using separately trained encoders alone and/or then combining them. Typically, BLIP and OpenCLIP models serve as reliable options for embedding data that involves both image and text modalities.

## Contributors

- [Kristóf Horváth, author](https://www.linkedin.com/in/kristof-horvath-0301/)
- [Mór Kapronczay, contributor](linkedin.com/in/mór-kapronczay-49447692)
- [Robert Turner, editor](https://robertturner.co/copyedit)
