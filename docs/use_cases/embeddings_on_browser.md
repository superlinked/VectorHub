<!-- TODO: Replace this text with a summary of article for SEO  test -->

# Vector Embeddings in the browser

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

![Visual Summary of our Tutorial](../assets/use_cases/embeddings_on_browser/embeddings-browser-animation)

---
## Our motivation

Let's say you want to create an intuitive semantic search application. We know a little about what we need: vector embeddings, maybe retrieval augmented generation. But how do operationalize our understanding in a real-world application? Don't we need a substantial hardware setup or expensive cloud APIs? Even if we had the requisite backend resources, who's going to set this up? Don't we also need highly specialized machine learning engineers or data scientists even to get started?

Happily, we're here to tell you that the answer to all of these concerns is No. 

You don't require high-end equipment or powerful GPUs. Thanks to pre-trained machine learning models, it's possible for any web technologist to create an intuitive semantic search application right within your browser, on a local machine, tailored to your data, without needing library installations or complex configurations for end-users. And you can start immediately. 

The following tutorial in creating a small-scale AI application demonstrates just how straightforward and efficient the process can be in a specific practical instance, but also more generally how you can operationalize vector embeddings for practical use cases.

Intrigued? Let's dive in!

# What we will be building

Our component combines React, TensorFlow.js, and Material-UI to provide a user-friendly interface for generating and visualizing sentence embeddings. We will be taking a user input text. We will split it into sentences, generating vector embeddings for each using TensorFlow.js. To assess the quality of our embeddings, we will provide functionality to generate the similarity between two sentences in the form of a similarity matrix. Our similarity matrix provides an overview between all pairs and renders this as a colorful heatmap. Our component manages all the necessary state and UI logic to enable this interaction.

Why should this matter to you? Imagine creating intuitive semantic search applications right within your web browser, tailored to your data! Our component makes this an immediate reality.

# In a nutshell

This is what we will be building:

1. We will import necessary dependencies such as React, Material-UI components, TensorFlow.js, and D3 for color interpolation.
2. The code defines a functional component named **`EmbeddingGenerator`,** a React functional component. This component represents the user interface for generating sentence embeddings and visualizing their similarity matrix.
3. We declare various state variables using the **`useState`** hook to manage user input, loading states, and results.
4. The **`handleSimilarityMatrix`** function toggles the display of the similarity matrix and calculates it when necessary.
5. The **`handleGenerateEmbedding`** function is responsible for starting the sentence embedding generation process. It splits the input sentences into individual sentences and triggers the **`embeddingGenerator`** function.
6. The **`calculateSimilarityMatrix`** function is a *memoized* function using the **`useCallback`** hook. It calculates the similarity matrix based on sentence embeddings.
7. The **`embeddingGenerator`** function is an asynchronous function that loads the Universal Sentence Encoder model and generates sentence embeddings.
8. We use the **`useEffect`** hook to render the similarity matrix as a colorful canvas when **`similarityMatrix`** changes.
9. The component's return statement defines the user interface, including input fields, buttons, and result displays.
10. The user input section includes a text area where the user can input sentences.
11. The embeddings output section displays the generated embeddings.
12. We provide two buttons. One generates the embeddings, and the other shows/hides the similarity matrix.
13. The code handles loading and model-loaded states, displaying loading indicators or model-loaded messages.
14. The similarity matrix section displays the colorful similarity matrix as a canvas when the user chooses to show it.

# Running a language model in the browser

The [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf) (Cer., et al., 2018) is a pre-trained machine learning model that converts text into vector representations. It takes sentences or paragraphs of text as input and produces fixed-length numeric vectors as output. These vectors effectively capture the meaning of the text, making them valuable for various natural language processing (NLP) tasks. For instance, we can gauge the similarity between two sentences by assessing the distance between their vector representations.

In our case, we'll utilize the 'Lite' version of this model, a scaled-down and faster variant of the full model. Despite its reduced size, it maintains strong performance while demanding less computational power. This feature makes it ideal for deployment in client-side code, mobile devices, or even directly within web browsers, eliminating the need for complex installations. Furthermore, it doesn't require a dedicated GPU, making it accessible to a broader range of users.

The rationale behind such models is straightforward. In many NLP applications, obtaining ample training data is a challenging endeavor. Data-hungry deep learning models are often infeasible due to this limitation, and annotating more supervised training data is an expensive solution. Consequently, most NLP projects in research and industry contexts can only access relatively small training datasets.

To address this constraint, many models employ pre-trained word embeddings like word2vec or GloVe, which transform individual words into vectors. However, recent developments have shown that pre-trained sentence-level embeddings can deliver impressive performance.

The Universal Sentence Encoder excels at generating embeddings for complete English sentences. When you input an English string, it generates a fixed-length vector embedding representing the entire sentence. These sentence embeddings are highly effective for computing semantic similarity between sentences and have demonstrated excellent performance in various semantic textual similarity benchmarks.

Moreover, these sentence embeddings are flexible and can be fine-tuned for specific tasks. Even with minimal task-specific training data, they can achieve surprisingly good results.

The model is built upon the transformer architecture, leveraging the attention mechanism to create context-aware representations for each word in a sentence. It carefully considers the order and identity of all other terms. These word representations are then combined into a fixed-length sentence vector. This combination is achieved by element-wise summation, followed by division by the square root of the sentence length. This normalization process prevents shorter sentences from dominating solely due to their brevity.

The primary objective is to create the encoder as a versatile tool suitable for various tasks. The model achieves this through multi-task learning, training a single model to support multiple downstream tasks.

In summary, the Universal Sentence Encoder excels at generating embeddings that represent the holistic meaning of sentences, even when dealing with limited training data, making it a valuable resource for NLP tasks.

# Our step-by-step tutorial

## Import modules
```tsx
import React, { FC, useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Typography,
  TextField,
  Paper,
  Button,
  CircularProgress,
} from "@mui/material";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-core';
import { interpolateGreens } from 'd3-scale-chromatic';
```

## State variables to manage user input, loading state, and results

These state variables are essential for managing user input, tracking loading states of machine learning models, and updating the user interface with the results, including the similarity matrix visualization.

Here, we define a series of state variables using the **`useState`** hook in a React functional component. These state variables are used to manage various aspects of the component's behavior and user interface.

```tsx
// State variables to manage user input, loading state, and results

// sentences - stores user input 
  const [sentences, setSentences] = useState<string>('');

// sentencesList - stores sentences split into array
  const [sentencesList, setSentencesList] = useState<string[]>([]);

// embeddings - stores generated embeddings
  const [embeddings, setEmbeddings] = useState<string>('');

// modelLoading - tracks model loading state
  const [modelLoading, setModelLoading] = useState(false);

// modelComputing - tracks model computing state
  const [modelComputing, setModelComputing] = useState(false);

// showSimilarityMatrix - controls matrix visibility
  const [showSimilarityMatrix, setShowSimilarityMatrix] = useState(false);

// embeddingModel - stores sentence embeddings  
  const [embeddingModel, setEmbeddingModel] = useState<tf.Tensor2D>(tf.tensor2d([[1, 2], [3, 4]]));

// canvasSize - size of canvas for matrix
  const [canvasSize, setCanvasSize] = useState(0);

// similarityMatrix - stores similarity matrix
  const [similarityMatrix, setSimilarityMatrix] = useState<number[][] | null>(null);
```

## Function to toggle the display of the similarity matrix

The **`handleSimilarityMatrix`** function toggles the display of a similarity matrix in the user interface by changing the **`showSimilarityMatrix`** state variable. If the matrix was previously shown, it hides it by setting it to **`null`**. If it wasn't shown, it calculates the matrix and sets it to be displayed in the user interface. This function is typically called when a user clicks a button or performs an action to show or hide the similarity matrix.

```tsx
// Toggles display of similarity matrix 
  const handleSimilarityMatrix = () => {
    // Toggle showSimilarityMatrix state
    setShowSimilarityMatrix(!showSimilarityMatrix);

    // If showing matrix, clear it 
    if (showSimilarityMatrix) {
      setSimilarityMatrix(null);

    // Else calculate and set matrix
    } else {
      const calculatedMatrix = calculateSimilarityMatrix(embeddingModel, sentencesList);
      setSimilarityMatrix(calculatedMatrix);
    }
  };
```

## Function to generate sentence embeddings and populate state

The **`handleGenerateEmbedding`** function is responsible for initiating the process of generating sentence embeddings. It sets the **`modelComputing`** state variable to **`true`** to indicate that the model is working, splits the user's input into individual sentences, updates the **`sentencesList`** state variable with these sentences, and then calls the **`embeddingGenerator`** function to start generating embeddings based on the individual sentences. This function is typically called when a user triggers the process, such as by clicking a "Generate Embedding" button. 

```tsx
// Generate embeddings for input sentences 
  const handleGenerateEmbedding = async () => {
    // Set model as computing
    setModelComputing(true);

    // Split input into individual sentences
    const individualSentences = sentences.split('.').filter(sentence => sentence.trim() !== '');

    // Save individual sentences to state
    setSentencesList(individualSentences);

    // Generate embeddings
    await embeddingGenerator(individualSentences);
  };
```

## Function to calculate the similarity matrix for sentence embeddings

In summary, the **`calculateSimilarityMatrix`** function computes a similarity matrix for a set of sentences by comparing the embeddings of each sentence with all other sentences. The matrix contains similarity scores for all possible sentence pairs and is used for further visualization or analysis.

This code snippet defines a JavaScript function named **`calculateSimilarityMatrix`**. This function is memoized using the **`useCallback`** hook, indicating that its behavior will remain consistent across renders unless its dependencies change.

```tsx
 // Calculates similarity matrix for sentence embeddings
  const calculateSimilarityMatrix = useCallback(

    // Embeddings and sentences arrays
    (embeddings: tf.Tensor2D, sentences: string[]) => {

      // Matrix to store scores
      const matrix = [];

      // Loop through each sentence
      for (let i = 0; i < sentences.length; i++) {

        // Row to store scores for sentence i
        const row = [];

        // Loop through each other sentence  
        for (let j = 0; j < sentences.length; j++) {

          // Get embeddings for sentences
          const sentenceI = tf.slice(embeddings, [i, 0], [1]);
          const sentenceJ = tf.slice(embeddings, [j, 0], [1]);

          const sentenceITranspose = false;
          const sentenceJTranspose = true;
          
          // Calculate similarity score
          const score = tf.matMul(sentenceI, sentenceJ, sentenceITranspose, sentenceJTranspose).dataSync();

          // Add score to row
          row.push(score[0]);
        }

        // Add row to matrix
        matrix.push(row);
      }

      // Return final matrix
      return matrix;
    },
    []
  );
```

## Function to generate sentence embeddings using the Universal Sentence Encoder

The **`embeddingGenerator`** function loads the Universal Sentence Encoder model, generates sentence embeddings for a list of sentences, and updates the component's state with the results. It also handles potential errors during the process. This function is typically called when a user triggers the embedding generation process, such as by clicking a "Generate Embedding" button.

```tsx
  // Generate embeddings using Universal Sentence Encoder (Cer., et al., 2018)
  const embeddingGenerator = useCallback(async (sentencesList: string[]) => {

    // Only run if model is not already loading
    if (!modelLoading) {
      try {

        // Set model as loading
        setModelLoading(true);

        // Load model
        const model = await use.load();

        // Array to store embeddings
        const sentenceEmbeddingArray: number[][] = [];

        // Generate embeddings
        const embeddings = await model.embed(sentencesList);

         // Process each sentence
        for (let i = 0; i < sentencesList.length; i++) {
          
          // Get embedding for sentence
          const sentenceI = tf.slice(embeddings, [i, 0], [1]);

          // Add to array
          sentenceEmbeddingArray.push(Array.from(sentenceI.dataSync()));
        }

        // Update embeddings state
        setEmbeddings(JSON.stringify(sentenceEmbeddingArray));

        // Update model state
        setEmbeddingModel(embeddings);

        // Reset loading states
        setModelLoading(false);
        setModelComputing(false);
      } catch (error) {
        console.error('Error loading model or generating embeddings:', error);

        // Handle errors
        setModelLoading(false);
        setModelComputing(false);
      }
    }
  }, [modelLoading]);
```

## useEffect hook to render the similarity matrix as a colorful canvas

This **`useEffect`** is triggered when the **`similarityMatrix`** or **`canvasSize`** changes. It draws a similarity matrix on an HTML canvas element. The matrix is represented as a grid of colored cells, with each color determined by the similarity value among sentences. This effect renders the visual representation of the similarity between sentences and is a dynamic part of the user interface.

```tsx
// Render similarity matrix as colored canvas
  useEffect(() => {

    // If matrix exists
    if (similarityMatrix) {

      // Get canvas element
      const canvas = document.querySelector('#similarity-matrix') as HTMLCanvasElement;

      // Set fixed canvas size
      setCanvasSize(250);

      // Set canvas dimensions
      canvas.width = canvasSize;
      canvas.height = canvasSize;

      // Get canvas context
      const ctx = canvas.getContext('2d');

      // If context available
      if (ctx) {

        // Calculate cell size
        const cellSize = canvasSize / similarityMatrix.length;

        // Loop through matrix
        for (let i = 0; i < similarityMatrix.length; i++) {
          for (let j = 0; j < similarityMatrix[i].length; j++) {

            // Set cell color based on value
            ctx.fillStyle = interpolateGreens(similarityMatrix[i][j]);

            // Draw cell  
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
          }
        }
      }
    }
  }, [similarityMatrix, canvasSize]);
```

## User Input Section

This code represents a part of the user interface where users can input multiple sentences. It includes a label, a multiline text input field, and the ability to control and update the input through React state management. The user's entered sentences are stored in the **`sentences`** state variable and can be used for further processing in the component.

```tsx
{/* User Input Section */}
      <Grid item md={6}>

        // Heading
        <Typography variant="h2" gutterBottom>Encode Sentences</Typography>

        // Multiline text input
        <TextField
          id="sentencesInput"
          label="Multiline"
          multiline
          fullWidth
          rows={6}
          value={sentences}
          onChange={(e) => setSentences(e.target.value)}
        />
      </Grid>
```

## Embeddings Output Section

A part of the user interface where generated sentence embeddings are displayed. It includes a label, a multiline text output field, and the ability to control and update the displayed content through React state management. The generated embeddings, stored in the **`embeddings`** state variable, are displayed to the user in this section.

```tsx
      {/* Embeddings Output Section */}
      <Grid item md={6}>

        // Heading
        <Typography variant="h2" gutterBottom>Embeddings</Typography>

        // Multiline text field to display embeddings
        <TextField
          id="embeddingOutput"
          label="Multiline"
          multiline
          fullWidth
          rows={6}
          value={embeddings}
          variant="filled"
        />
      </Grid>
```

## Generate Embedding Button

This code represents a button in the user interface that users can click to trigger the generation of sentence embeddings. The button is styled as a raised, solid button, and it is initially disabled if there are no input sentences (**`!sentences`**) or if the model is currently loading (**`modelLoading`**). When clicked, it invokes the **`handleGenerateEmbedding`** function to initiate the embedding generation process.

```tsx
      {/* Generate Embedding Button */}
      <Grid item xs={6}>
        <Button
          variant="contained"
          id='generate-embedding'
          onClick={handleGenerateEmbedding}
          disabled={!sentences || modelLoading}
        >
          Generate Embedding
        </Button>
      </Grid>
```

## Model Indicator

This code controls what is displayed in the user interface based on the values of the **`modelComputing`** and **`modelLoading`** state variables. If  is **`true`**, it first checks if . If it is, a loading indicator is displayed. If **`false`**, a message indicating that the model is loaded is shown. If , nothing is rendered in this section. This conditional rendering allows the user to see either a loading indicator or a model loaded message based on the status of model loading and computing.

```tsx
      {/* Display model loading or loaded message */}
      {modelComputing ? (
        modelLoading ? (

          // Show loading indicator
          <Grid item xs={12}>
            <Box>
              <CircularProgress />
              <Typography variant="body1">Loading the model...</Typography>
            </Box>
          </Grid>
        ) : (

          // Show model loaded message 
          <Grid container>
            <Grid item xs={12}>
              <Box>
                <Typography variant="body1">Model Loaded</Typography>
              </Box>
            </Grid>
          </Grid>

        )
      ) : null}
```

## Similarity Matrix
This code controls the rendering of the similarity matrix section of the user interface based on the value of the **`showSimilarityMatrix`** state variable. If it is **`true`**, a section containing the similarity matrix is displayed. The section includes a title, "Similarity Matrix," and a canvas element for rendering the matrix. If **`false`**, nothing is rendered in this section, providing a way to show or hide the similarity matrix in the user interface.

```tsx
{/* Similarity Matrix Section  */}
      {showSimilarityMatrix ? (
        
        <Grid item xs={12}>

          <Paper elevation={3}>

            <Typography variant="h3">
            Similarity Matrix
            </Typography>

            <canvas 
            id="similarity-matrix" 
            width={canvasSize} 
            height={canvasSize} 
            style={{ width: '100%', height: '100%' }}>
            </canvas>

          </Paper>

        </Grid>
        
      ) : null}
    </>
  );
};
```

## Taking our component for a ride

How can we determine the effectiveness of our component and the quality of our model's vector embeddings? Evaluating the component's functionality is straightforward – we run and test it. However, assessing the quality of the embeddings is a bit more complex, as we are dealing with arrays of 512 elements. It begs the question of how to gauge their effectiveness. Here is where the similarity matrix comes into play.
We employ the dot product between vectors for each pair of sentences to discern their proximity or dissimilarity. To illustrate this, let's take two random pages from Wikipedia, each containing different paragraphs. They will provide us with a total of seven sentences for comparison.

[Los Angeles Herald](https://en.wikipedia.org/wiki/Los_Angeles_Herald)

[The quick brown fox jumps over the lazy dog](https://en.wikipedia.org/wiki/The_quick_brown_fox_jumps_over_the_lazy_dog)

### Paragraph 1

> "The quick brown fox jumps over the lazy dog" is an English-language pangram – a sentence that contains all the letters of the alphabet. The phrase is commonly used for touch-typing practice, testing typewriters and computer keyboards, displaying examples of fonts, and other applications involving text where the use of all letters in the alphabet is desired.
> 

### Paragraph 2

> The Los Angeles Herald or the Evening Herald was a newspaper published in Los Angeles in the late 19th and early 20th centuries. Founded in 1873 by Charles A. Storke, the newspaper was acquired by William Randolph Hearst in 1931. It merged with the Los Angeles Express and became an evening newspaper known as the Los Angeles Herald-Express. A 1962 combination with Hearst's morning Los Angeles Examiner resulted in its final incarnation as the evening Los Angeles Herald-Examiner.
> 

Once we input these sentences into our model and generate the similarity matrix, something remarkable happens. Sentences from the same paragraphs exhibit a close resemblance, marked by a dark green hue. You can even observe how they cluster together, forming two distinct squares. Conversely, sentences from different paragraphs display little similarity, represented by a light green color. 

This process allows us to rapidly construct an in-browser vector embedding generator that we can readily apply to real-world tasks. 

What's even more remarkable is that we don't need to rely on cloud models or expensive hardware. Everything operates seamlessly within the browser, thanks to web development libraries and our programming language, TypeScript.

## Similarity Matrix for our sentences

![Similarity Matrix for seven sentences from two documents](../assets/use_cases/embeddings_on_browser/ embeddings-browser-similarity-matrix.png)

## Contributors

- [Rod Rivera](http://twitter.com/rorcde)
