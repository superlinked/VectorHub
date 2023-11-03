<!-- TODO: Replace this text with a summary of article for SEO -->

# Vector Embeddings in the browser

<!-- TODO: Cover image: 
1. You can create your own cover image and put it in the correct asset directory,
2. or you can give an explanation on how it should be and we will help you create one. Please tag arunesh@superlinked.com or @AruneshSingh (GitHub) in this case. -->

![Visual Summary of our Tutorial](../assets/use_cases/embeddings_on_browser/embeddings-browser-animation)

---
## Contributors

- [Rod Rivera](http://twitter.com/rorcde)


# Our motivation

We often encounter terms like vector embeddings, RAGs, and other technical jargon, but it's challenging to grasp how these concepts work in real-world applications. Additionally, delving into these areas often appears cost-prohibitive, involving substantial investments in hardware or the expense of utilizing cloud APIs. It might lead to the assumption that you'd need highly specialized machine learning engineers or data scientists to make any headway.

However, today, we'll discover that this isn't the case. Working with vector embeddings can be accessible to anyone involved in web technologies, thanks to the capabilities of pre-trained machine learning models. What's more, our tutorial can be executed entirely on a local machine without the need for library installations or complex configurations for end-users. The best part is that you don't require high-end equipment or powerful GPUs.

This tutorial not only guides us in creating a practical, small-scale AI application but also enhances our understanding of vector embeddings in practical scenarios.

Intrigued? Let's dive in and explore how to make it happen!

# What we will be building

Our component combines React, TensorFlow.js, and Material-UI to provide a user-friendly interface for generating and visualizing sentence embeddings. We will be taking a user input text. We will split it into sentences, generating vector embeddings for each using TensorFlow.js. To assess the quality of our embeddings, we will provide functionality to generate the similarity between two sentences in the form of a similarity matrix. Our similarity matrix provides an overview between all pairs and renders this as a colorful heatmap. Our component manages all the necessary state and UI logic to enable this interaction.

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

# The model that we will be using

The Universal Sentence Encoder is a pre-trained machine learning model that converts text into vector representations. It takes sentences or paragraphs of text as input and produces fixed-length numeric vectors as output. These vectors effectively capture the meaning of the text, making them valuable for various natural language processing (NLP) tasks. For instance, we can gauge the similarity between two sentences by assessing the distance between their vector representations.

In our case, we'll utilize the 'Lite' version of this model, a scaled-down and faster variant of the full model. Despite its reduced size, it maintains strong performance while demanding less computational power. This feature makes it ideal for deployment in client-side code, mobile devices, or even directly within web browsers, eliminating the need for complex installations. Furthermore, it doesn't require a dedicated GPU, making it accessible to a broader range of users.

The rationale behind such models is straightforward. In many NLP applications, obtaining ample training data is a challenging endeavor. Data-hungry deep learning models are often infeasible due to this limitation, and annotating more supervised training data is an expensive solution. Consequently, most NLP projects in research and industry contexts can only access relatively small training datasets.

To address this constraint, many models employ pre-trained word embeddings like word2vec or GloVe, which transform individual words into vectors. However, recent developments have shown that pre-trained sentence-level embeddings can deliver impressive performance.

The Universal Sentence Encoder excels at generating embeddings for complete English sentences. When you input an English string, it generates a fixed-length vector embedding representing the entire sentence. These sentence embeddings are highly effective for computing semantic similarity between sentences and have demonstrated excellent performance in various semantic textual similarity benchmarks.

Moreover, these sentence embeddings are flexible and can be fine-tuned for specific tasks. Even with minimal task-specific training data, they can achieve surprisingly good results.

The model is built upon the transformer architecture, leveraging the attention mechanism to create context-aware representations for each word in a sentence. It carefully considers the order and identity of all other terms. These word representations are then combined into a fixed-length sentence vector. This combination is achieved by element-wise summation, followed by division by the square root of the sentence length. This normalization process prevents shorter sentences from dominating solely due to their brevity.

The primary objective is to create the encoder as a versatile tool suitable for various tasks. The model achieves this through multi-task learning, training a single model to support multiple downstream tasks.

In summary, the Universal Sentence Encoder excels at generating embeddings that represent the holistic meaning of sentences, even when dealing with limited training data, making it a valuable resource for NLP tasks.

[](https://arxiv.org/pdf/1803.11175.pdf)

# Our step-by-step tutorial

## Import modules

1. **`import React, { FC, useState, useEffect, useCallback } from 'react';`**:
    - This line imports necessary modules from the 'react' library.
    - **`React`** is the core library for building user interfaces in React.
    - **`FC`** is a TypeScript type definition, representing a functional component.
    - **`useState`**, **`useEffect`**, and **`useCallback`** are React hooks that allow you to manage component state and side effects.
2. **`import { Box, Grid, Typography, TextField, Paper, Button, CircularProgress } from "@mui/material";`**:
    - This line imports various Material-UI components from the **`@mui/material`** library.
    - Material-UI is a popular UI framework for React applications, providing pre-styled components for building user interfaces.
3. **`import '@tensorflow/tfjs-backend-cpu';`** and **`import '@tensorflow/tfjs-backend-webgl';`**:
    - These lines import the TensorFlow.js backend modules for CPU and WebGL.
    - TensorFlow.js is a JavaScript library that allows machine learning and deep learning models to run in the browser.
4. **`import * as use from '@tensorflow-models/universal-sentence-encoder';`**:
    - This line imports the Universal Sentence Encoder model from TensorFlow.js.
    - The Universal Sentence Encoder is a pre-trained model that encodes sentences into numerical vectors, allowing similarity comparisons between sentences.
5. **`import * as tf from '@tensorflow/tfjs-core';`**:
    - This line imports the core TensorFlow.js library.
    - TensorFlow.js core provides fundamental functionalities for machine learning in JavaScript.
6. **`import { interpolateGreens } from 'd3-scale-chromatic';`**:
    - This line imports the **`interpolateGreens`** function from the 'd3-scale-chromatic' library.
    - D3 is a data visualization library, and **`interpolateGreens`** is used to interpolate colors in the green color scale.

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

Here, we define a series of state variables using the **`useState`** hook in a React functional component. These state variables are used to manage various aspects of the component's behavior and user interface:

1. **`sentences`** and **`setSentences`**:
    - **`sentences`** is a state variable used to store the user's input sentences. It is initialized with an empty string.
    - **`setSentences`** is a function to update the **`sentences`** state variable.
2. **`sentencesList`** and **`setSentencesList`**:
    - **`sentencesList`** is a state variable that stores the user's input sentences split into an array of individual sentences.
    - It is initialized as an empty array.
    - **`setSentencesList`** is a function to update the **`sentencesList`** state variable.
3. **`embeddings`** and **`setEmbeddings`**:
    - **`embeddings`** is a state variable used to store the generated sentence embeddings.
    - It is initialized with an empty string.
    - **`setEmbeddings`** is a function to update the **`embeddings`** state variable.
4. **`modelLoading`** and **`setModelLoading`**:
    - **`modelLoading`** is a state variable that tracks whether a machine learning model is being loaded.
    - It is initially set to **`false`**.
    - **`setModelLoading`** is a function to update the **`modelLoading`** state variable.
5. **`modelComputing`** and **`setModelComputing`**:
    - **`modelComputing`** is a state variable that tracks whether the model is currently computing embeddings.
    - It is initially set to **`false`**.
    - **`setModelComputing`** is a function to update the **`modelComputing`** state variable.
6. **`showSimilarityMatrix`** and **`setShowSimilarityMatrix`**:
    - **`showSimilarityMatrix`** is a state variable that controls the visibility of a similarity matrix in the user interface.
    - It is initially set to **`false`**.
    - **`setShowSimilarityMatrix`** is a function to update the **`showSimilarityMatrix`** state variable.
7. **`embeddingModel`** and **`setEmbeddingModel`**:
    - **`embeddingModel`** is a state variable that stores a TensorFlow.js tensor. It is used to represent the embeddings of sentences.
    - It is initialized with a simple tensor with values **`[[1, 2], [3, 4]]`**.
    - **`setEmbeddingModel`** is a function to update the **`embeddingModel`** state variable.
8. **`canvasSize`** and **`setCanvasSize`**:
    - **`canvasSize`** is a state variable representing the size of a canvas used to display the similarity matrix.
    - It is initially set to **`0`**.
    - **`setCanvasSize`** is a function to update the **`canvasSize`** state variable.
9. **`similarityMatrix`** and **`setSimilarityMatrix`**:
    - **`similarityMatrix`** is a state variable that stores the calculated similarity matrix for sentence embeddings.
    - It is initially set to **`null`**.
    - **`setSimilarityMatrix`** is a function to update the **`similarityMatrix`** state variable.

```tsx
// State variables to manage user input, loading state, and results
  const [sentences, setSentences] = useState<string>('');
  const [sentencesList, setSentencesList] = useState<string[]>([]);
  const [embeddings, setEmbeddings] = useState<string>('');
  const [modelLoading, setModelLoading] = useState(false);
  const [modelComputing, setModelComputing] = useState(false);
  const [showSimilarityMatrix, setShowSimilarityMatrix] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState<tf.Tensor2D>(tf.tensor2d([[1, 2], [3, 4]]));
  const [canvasSize, setCanvasSize] = useState(0);
  const [similarityMatrix, setSimilarityMatrix] = useState<number[][] | null>(null);
```

## Function to toggle the display of the similarity matrix

The **`handleSimilarityMatrix`** function toggles the display of a similarity matrix in the user interface by changing the **`showSimilarityMatrix`** state variable. If the matrix was previously shown, it hides it by setting it to **`null`**. If it wasn't shown, it calculates the matrix and sets it to be displayed in the user interface. This function is typically called when a user clicks a button or performs an action to show or hide the similarity matrix.

1. **`const handleSimilarityMatrix = () => { ... }`**:
    - This line declares a function named **`handleSimilarityMatrix`** using the arrow function syntax.
2. **`setShowSimilarityMatrix(!showSimilarityMatrix);`**:
    - This line toggles the **`showSimilarityMatrix`** state variable. It negates its current value (if it's **`true`**, it becomes **`false`**, and vice versa) to switch between showing and hiding the similarity matrix in the user interface.
3. **`if (showSimilarityMatrix) { ... }`**:
    - This is an **`if`** statement that checks the value of the **`showSimilarityMatrix`** state variable. If it is currently **`true`**, it executes the code block inside this block.
4. **`setSimilarityMatrix(null);`**:
    - If **`showSimilarityMatrix`** was previously **`true`**, this line sets the **`similarityMatrix`** state variable to **`null`**. This effectively hides the similarity matrix by clearing its data.
5. **`else { ... }`**:
    - If **`showSimilarityMatrix`** was not **`true (i.e., it was`** false`), this code block is executed.
6. **`const calculatedMatrix = calculateSimilarityMatrix(embeddingModel, sentencesList);`**:
    - Here, the code calls a function **`calculateSimilarityMatrix`** and passes the **`embeddingModel`** and **`sentencesList`** as arguments. This function is used to calculate the similarity matrix for sentence embeddings.
7. **`setSimilarityMatrix(calculatedMatrix);`**:
    - After calculating the similarity matrix, the result is set into the **`similarityMatrix`** state variable. This updates the state with the calculated data, effectively showing the similarity matrix in the user interface.
    

```tsx
// Function to toggle the display of the similarity matrix
  const handleSimilarityMatrix = () => {
    setShowSimilarityMatrix(!showSimilarityMatrix);
    if (showSimilarityMatrix) {
      setSimilarityMatrix(null);
    } else {
      const calculatedMatrix = calculateSimilarityMatrix(embeddingModel, sentencesList);
      setSimilarityMatrix(calculatedMatrix);
    }
  };
```

## Function to generate sentence embeddings and populate state

The **`handleGenerateEmbedding`** function is responsible for initiating the process of generating sentence embeddings. It sets the **`modelComputing`** state variable to **`true`** to indicate that the model is working, splits the user's input into individual sentences, updates the **`sentencesList`** state variable with these sentences, and then calls the **`embeddingGenerator`** function to start generating embeddings based on the individual sentences. This function is typically called when a user triggers the process, such as by clicking a "Generate Embedding" button. 

1. **`const handleGenerateEmbedding = async () => { ... }`**:
    - This line declares an asynchronous function named **`handleGenerateEmbedding`** using the arrow function syntax, indicating that it might perform asynchronous operations.
2. **`setModelComputing(true);`**:
    - This line sets the **`modelComputing`** state variable to **`true`**, indicating that the model is currently computing embeddings.
3. **`const individualSentences = sentences.split('.').filter(sentence => sentence.trim() !== '');`**:
    - This code splits the **`sentences`** state variable into individual sentences using the period (.) as a delimiter.
    - It then filters out any empty or whitespace-only sentences using the **`filter`** method.
    - The result is stored in the **`individualSentences`** variable, which becomes an array of individual sentences.
4. **`setSentencesList(individualSentences);`**:
    - This line updates the **`sentencesList`** state variable with the array of individual sentences obtained in the previous step. This state variable now holds a list of sentences that the model will generate embeddings for.
5. **`await embeddingGenerator(individualSentences);`**:
    - This line calls the **`embeddingGenerator`** function, passing the **`individualSentences`** array as an argument.
    - The **`await`** keyword indicates that this operation is asynchronous, and it waits for the **`embeddingGenerator`** function to complete before moving on to the next step.

```tsx
// Function to generate sentence embeddings and populate state
  const handleGenerateEmbedding = async () => {
    // Start the model computing
    setModelComputing(true);
    const individualSentences = sentences.split('.').filter(sentence => sentence.trim() !== '');
    // Split the input sentences into individual sentences
    setSentencesList(individualSentences);
    await embeddingGenerator(individualSentences);
  };
```

## Function to calculate the similarity matrix for sentence embeddings

In summary, the **`calculateSimilarityMatrix`** function computes a similarity matrix for a set of sentences by comparing the embeddings of each sentence with all other sentences. The matrix contains similarity scores for all possible sentence pairs and is used for further visualization or analysis.

This code snippet defines a JavaScript function named **`calculateSimilarityMatrix`**. This function is memoized using the **`useCallback`** hook, indicating that its behavior will remain consistent across renders unless its dependencies change. Let's break down the code in detail:

1. **`const calculateSimilarityMatrix = useCallback(...)`**:
    - This line defines the **`calculateSimilarityMatrix`** function and memoizes it using the **`useCallback`** hook. Memoization helps optimize the performance of the function by preventing unnecessary re-calculations.
2. **`(embeddings: tf.Tensor2D, sentences: string[]) => { ... }`**:
    - The function takes two parameters:
        - **`embeddings`**: A TensorFlow.js **`tf.Tensor2D`** representing sentence embeddings.
        - **`sentences`**: An array of strings, each containing a sentence for which the similarity score will be calculated.
3. **`const matrix = [];`**:
    - This line initializes an empty array called **`matrix`**. This array will store the similarity scores between sentences.
4. **`for (let i = 0; i < sentences.length; i++) { ... }`**:
    - This is a nested **`for`** loop that iterates over each pair of sentences to calculate their similarity scores.
5. **`const row = [];`**:
    - Inside the outer loop, a new empty array called **`row`** is created. This array will hold the similarity scores for a specific sentence against all other sentences.
6. **`for (let j = 0; j < sentences.length; j++) { ... }`**:
    - Inside the inner loop, the function iterates over all sentences again to compare each sentence (i) with all other sentences (j).
7. **`const sentenceI = tf.slice(embeddings, [i, 0], [1]);`**:
    - This line extracts a single row from the **`embeddings`** tensor. **`sentenceI`** now contains the embeddings for the i-th sentence.
8. **`const sentenceJ = tf.slice(embeddings, [j, 0], [1]);`**:
    - Similarly, this line extracts a single row from the **`embeddings`** tensor for the j-th sentence.
9. **`const sentenceITranspose = false;`** and **`const sentenceJTranspose = true;`**:
    - These variables indicate whether to transpose (flip) the rows and columns of the tensors when performing matrix multiplication. In this case, **`sentenceI`** remains untransposed, and **`sentenceJ`** is transposed.
10. **`const score = tf.matMul(sentenceI, sentenceJ, sentenceITranspose, sentenceJTranspose).dataSync();`**:
    - This line calculates the similarity score between **`sentenceI`** and **`sentenceJ`** using matrix multiplication. The **`dataSync`** method retrieves the numerical value of the score.
11. **`row.push(score[0]);`**:
    - The calculated similarity score is added to the **`row`** array.
12. **`matrix.push(row);`**:
    - After processing all sentences against a specific sentence (i), the **`row`** array, which contains similarity scores for that sentence, is added to the **`matrix`** array.
13. **`return matrix;`**:
    - The **`matrix`** array, containing the similarity scores between all pairs of sentences, is returned as the result of the function.

```tsx
  // Function to calculate the similarity matrix for sentence embeddings
  const calculateSimilarityMatrix = useCallback(
    (embeddings: tf.Tensor2D, sentences: string[]) => {
      const matrix = [];

      for (let i = 0; i < sentences.length; i++) {
        const row = [];
        for (let j = 0; j < sentences.length; j++) {
          const sentenceI = tf.slice(embeddings, [i, 0], [1]);
          const sentenceJ = tf.slice(embeddings, [j, 0], [1]);
          const sentenceITranspose = false;
          const sentenceJTranspose = true;
          const score = tf.matMul(sentenceI, sentenceJ, sentenceITranspose, sentenceJTranspose).dataSync();
          row.push(score[0]);
        }
        matrix.push(row);
      }

      return matrix;
    },
    []
  );
```

## Function to generate sentence embeddings using Universal Sentence Encoder (Cer., et al., 2018)

The **`embeddingGenerator`** function loads the Universal Sentence Encoder model, generates sentence embeddings for a list of sentences, and updates the component's state with the results. It also handles potential errors during the process. This function is typically called when a user triggers the embedding generation process, such as by clicking a "Generate Embedding" button. Let's break down the code in detail:

1. **`const embeddingGenerator = useCallback(async (sentencesList: string[]) => { ... }, [modelLoading]);`**:
    - This line defines the **`embeddingGenerator`** function and memoizes it using the **`useCallback`** hook.
    - The function is asynchronous (**`async`**) as it performs operations that involve loading and using machine learning models.
    - It takes an array of **`sentencesList`** as a parameter, representing the list of sentences for which embeddings will be generated.
    - The **`[modelLoading]`** dependency array indicates that this function depends on the **`modelLoading`** state variable. This means it will be recreated only when **`modelLoading`** changes.
2. **`if (!modelLoading) { ... }`**:
    - This condition checks if the **`modelLoading`** state variable is **`false`** to ensure that the model is not currently being loaded.
3. **`try { ... } catch (error) { ... }`**:
    - This code is enclosed in a **`try-catch`** block to handle any errors that might occur during the function's execution.
4. **`setModelLoading(true);`**:
    - This line sets the **`modelLoading`** state variable to **`true`** to indicate that the model loading process has started.
5. **`const model = await use.load();`**:
    - This line loads the Universal Sentence Encoder model using the **`use.load()`** method. The **`await`** keyword indicates that the code will wait for the model to be loaded before proceeding.
6. **`const sentenceEmbeddingArray: number[][] = [];`**:
    - An empty array called **`sentenceEmbeddingArray`** is declared. This array will store the generated sentence embeddings.
7. **`const embeddings = await model.embed(sentencesList);`**:
    - This line generates sentence embeddings for the provided **`sentencesList`** using the loaded Universal Sentence Encoder model.
8. **`for (let i = 0; i < sentencesList.length; i++) { ... }`**:
    - A **`for`** loop iterates over each sentence in **`sentencesList`** to process them individually.
9. **`const sentenceI = tf.slice(embeddings, [i, 0], [1]);`**:
    - This line extracts the embedding for the i-th sentence from the **`embeddings`** using TensorFlow.js. The **`sentenceI`** variable now holds the embedding for a single sentence.
10. **`sentenceEmbeddingArray.push(Array.from(sentenceI.dataSync()));`**:
    - The embedding for the sentence is converted to an array of numbers and added to the **`sentenceEmbeddingArray`**. This array will contain the embeddings for all sentences.
11. **`setEmbeddings(JSON.stringify(sentenceEmbeddingArray));`**:
    - The **`sentenceEmbeddingArray`** is converted to a JSON string and set as the value of the **`embeddings`** state variable. This updates the component's state with the generated embeddings.
12. **`setEmbeddingModel(embeddings);`**:
    - The **`embeddings`** obtained from the Universal Sentence Encoder are set as the value of the **`embeddingModel`** state variable.
13. **`setModelLoading(false);`** and **`setModelComputing(false);`**:
    - Both **`modelLoading`** and **`modelComputing`** state variables are set to **`false`** to indicate that the model loading and computation have been completed.
14. **`catch (error) { ... }`**:
    - In case of any errors during the process, this block handles the error.
    - An error message is logged to the console.
    - Both **`modelLoading`** and **`modelComputing`** state variables are set to **`false`** to handle errors and reset the loading and computing states.

```tsx

  // Function to generate sentence embeddings using Universal Sentence Encoder (Cer., et al., 2018)
  // https://arxiv.org/pdf/1803.11175.pdf
  const embeddingGenerator = useCallback(async (sentencesList: string[]) => {
    if (!modelLoading) {
      try {
        setModelLoading(true);
        const model = await use.load();
        const sentenceEmbeddingArray: number[][] = [];
        const embeddings = await model.embed(sentencesList);
        for (let i = 0; i < sentencesList.length; i++) {
          const sentenceI = tf.slice(embeddings, [i, 0], [1]);
          sentenceEmbeddingArray.push(Array.from(sentenceI.dataSync()));
        }
        setEmbeddings(JSON.stringify(sentenceEmbeddingArray));
        setEmbeddingModel(embeddings);
        setModelLoading(false);
        setModelComputing(false);
      } catch (error) {
        console.error('Error loading model or generating embeddings:', error);
        setModelLoading(false);
        setModelComputing(false);
      }
    }
  }, [modelLoading]);
```

## useEffect hook to render the similarity matrix as a colorful canvas

This **`useEffect`** is triggered when the **`similarityMatrix`** or **`canvasSize`** changes. It draws a similarity matrix on an HTML canvas element. The matrix is represented as a grid of colored cells, with each color determined by the similarity value among sentences. This effect renders the visual representation of the similarity between sentences and is a dynamic part of the user interface.

1. **`useEffect(() => { ... }, [similarityMatrix, canvasSize]);`**:
    - The **`useEffect`** hook is used to perform side effects in a React component.
    - It takes a function (the effect) and an array of dependencies.
    - The effect is executed when the dependencies in the array change. In this case, it runs when **`similarityMatrix`** or **`canvasSize`** changes.
2. **`if (similarityMatrix) { ... }`**:
    - This condition checks if the **`similarityMatrix`** state variable is not null or undefined. If it has a value (i.e., it exists), the code within the block is executed.
3. **`const canvas = document.querySelector('#similarity-matrix') as HTMLCanvasElement;`**:
    - This line attempts to select an HTML canvas element with the ID 'similarity-matrix' using the **`document.querySelector`** method.
    - The selected canvas is cast to the type **`HTMLCanvasElement`**.
4. **`setCanvasSize(250);`**:
    - This line updates the **`canvasSize`** state variable with a fixed value of 250. This value represents the width and height of the canvas.
5. **`canvas.width = canvasSize;`** and **`canvas.height = canvasSize;`**:
    - These lines set the width and height of the canvas element to the value stored in the **`canvasSize`** state variable.
6. **`const ctx = canvas.getContext('2d');`**:
    - This line obtains the 2D rendering context of the canvas, which is necessary for drawing on it.
7. **`if (ctx) { ... }`**:
    - This condition checks if the rendering context (**`ctx`**) is available and not null.
8. **`const cellSize = canvasSize / similarityMatrix.length;`**:
    - This line calculates the size of each cell in the similarity matrix by dividing the **`canvasSize`** by the length of the **`similarityMatrix`**. This ensures that the entire matrix fits within the canvas.
9. **`for (let i = 0; i < similarityMatrix.length; i++) { ... }`**:
    - This is a nested **`for`** loop that iterates over the rows of the **`similarityMatrix`**.
10. **`for (let j = 0; j < similarityMatrix[i].length; j++) { ... }`**:
    - Inside the inner loop, the code iterates over the columns of the similarity matrix for a given row (i).
11. **`ctx.fillStyle = interpolateGreens(similarityMatrix[i][j]);`**:
    - This line sets the fill color of the drawing context (**`ctx`**) to a color determined by the **`interpolateGreens`** function. The color is based on the similarity value found in the **`similarityMatrix`**.
12. **`ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);`**:
    - This line draws a filled rectangle (cell) on the canvas at the position determined by the current row (**`i`**) and column (**`j`**). The dimensions of the cell are set by **`cellSize`**.

```tsx
// useEffect hook to render the similarity matrix as a colorful canvas
  useEffect(() => {
    if (similarityMatrix) {
      const canvas = document.querySelector('#similarity-matrix') as HTMLCanvasElement;
      setCanvasSize(250);
      canvas.width = canvasSize;
      canvas.height = canvasSize;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const cellSize = canvasSize / similarityMatrix.length;
        for (let i = 0; i < similarityMatrix.length; i++) {
          for (let j = 0; j < similarityMatrix[i].length; j++) {
            ctx.fillStyle = interpolateGreens(similarityMatrix[i][j]);
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
          }
        }
      }
    }
  }, [similarityMatrix, canvasSize]);
```

## User Input Section

This code represents a part of the user interface where users can input multiple sentences. It includes a label, a multiline text input field, and the ability to control and update the input through React state management. The user's entered sentences are stored in the **`sentences`** state variable and can be used for further processing in the component.

1. **`<Grid item md={6}>`**:
    - This is a JSX element from the Material-UI library (**`<Grid>`**). It's used for layout purposes and defines a grid item that occupies 6 out of 12 available columns on a responsive grid system.
    - The **`md={6}`** property specifies that on medium-sized screens and larger, this grid item should take up six columns, effectively splitting the available width in half.
2. **`<Typography variant="h2" gutterBottom>Encode Sentences</Typography>`**:
    - This line renders a Material-UI **`<Typography>`** component with the text "Encode Sentences."
    - The **`variant="h2"`** prop sets the typography style to an "h2" heading, making the text larger and more prominent.
    - The **`gutterBottom`** prop adds some spacing at the bottom of the text for visual separation.
3. **`<TextField id="sentencesInput" label="Multiline" multiline fullWidth rows={6} value={sentences} onChange={(e) => setSentences(e.target.value)} />`**:
    - This JSX element renders a Material-UI **`<TextField>`** component for user input.
    - **`id="sentencesInput"`** sets the unique ID of the input field, which can be used for referencing and styling.
    - **`label="Multiline"`** specifies the label to be displayed above the input field. In this case, it's labeled as "Multiline."
    - **`multiline`** indicates that this text field should allow multiple lines of text to be entered.
    - **`fullWidth`** makes the text field occupy the full width of its container.
    - **`rows={6}`** sets the initial number of rows for the multiline input, indicating that it can display six lines of text.
    - **`value={sentences}`** sets the value of the input field. The **`sentences`** variable is used as the value, controlled by the component's state.
    - **`onChange={(e) => setSentences(e.target.value)`** attaches an event handler to the input field. When the user types or modifies the content, this handler updates the **`sentences`** state variable with the new input value.

```tsx
 {/* User Input Section */}
      <Grid item md={6}>
        <Typography variant="h2" gutterBottom>Encode Sentences</Typography>
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

1. **`<Grid item md={6}>`**:
    - This is a JSX element from the Material-UI library (**`<Grid>`**). It's used for layout purposes and defines a grid item that occupies 6 out of 12 available columns on a responsive grid system.
    - The **`md={6}`** property specifies that on medium-sized screens and larger, this grid item should take up 6 columns, effectively splitting the available width in half.
2. **`<Typography variant="h2" gutterBottom>Embeddings</Typography>`**:
    - This line renders a Material-UI **`<Typography>`** component with the text "Embeddings."
    - The **`variant="h2"`** prop sets the typography style to an "h2" heading, making the text larger and more prominent.
    - The **`gutterBottom`** prop adds some spacing at the bottom of the text for visual separation.
3. **`<TextField id="embeddingOutput" label="Multiline" multiline fullWidth rows={6} value={embeddings} variant="filled" />`**:
    - This JSX element renders a Material-UI **`<TextField>`** component for displaying the generated sentence embeddings.
    - **`id="embeddingOutput"`** sets the unique ID of the text field, which can be used for referencing and styling.
    - **`label="Multiline"`** specifies the label to be displayed above the text field. In this case, it's labeled as "Multiline."
    - **`multiline`** indicates that this text field allows multiple lines of text to be displayed.
    - **`fullWidth`** makes the text field occupy the full width of its container.
    - **`rows={6}`** sets the initial number of rows for the multiline output, indicating that it can display six lines of text.
    - **`value={embeddings}`** sets the value of the text field. The **`embeddings`** variable is used as the value, controlled by the component's state.
    - **`variant="filled"`** changes the visual style of the text field to a "filled" variant, which typically shows the text field with a filled background.

```tsx
{/* Embeddings Output Section */}
      <Grid item md={6}>
        <Typography variant="h2" gutterBottom>Embeddings</Typography>
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

1. **`<Grid item xs={6}>`**:
    - This is a JSX element from the Material-UI library (**`<Grid>`**). It's used for layout purposes and defines a grid item that occupies 6 out of 12 available columns on a responsive grid system.
    - The **`xs={6}`** property specifies that on extra-small screens and larger, this grid item should take up 6 columns, effectively splitting the available width in half.
2. **`<Button variant="contained" id='generate-embedding' onClick={handleGenerateEmbedding} disabled={!sentences || modelLoading}>Generate Embedding</Button>`**:
    - This JSX element renders a Material-UI **`<Button>`** component that serves as the "Generate Embedding" button.
    - **`variant="contained"`** sets the visual style of the button to "contained," meaning it will have a solid background and is visually raised.
    - **`id='generate-embedding'`** sets the unique ID for the button, which can be used for referencing and styling.
    - **`onClick={handleGenerateEmbedding}`** attaches an event handler function, **`handleGenerateEmbedding`**, to the button. When the user clicks the button, this function will be executed.
    - **`disabled={!sentences || modelLoading}`** sets the button's initial disabled state. It is disabled if either of the following conditions is met:
        - **`!sentences`**: If the **`sentences`** state variable is falsy (empty or undefined), indicating no input.
        - **`modelLoading`**: If the **`modelLoading`** state variable is **`true`**, it indicates that the model is currently loading.
3. **`"Generate Embedding"`**:
    - The text "Generate Embedding" is the label displayed on the button.

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

1. **`{modelComputing ? ... : null}`**:
    - This is a conditional statement that checks the value of the **`modelComputing`** state variable. If **`modelComputing`** is **`true`**, the code block inside the first set of parentheses will be executed. If it's **`false`**, **`null`** will be returned, indicating that nothing should be rendered.
2. **`modelLoading ? (...) : (...)`**:
    - This is a nested conditional statement. It checks the value of the **`modelLoading`** state variable.
    - If **`modelLoading`** is **`true`**, the code block inside the first set of parentheses is executed. This part is responsible for rendering a loading indicator.
    - If **`modelLoading`** is **`false`**, the code block inside the second set of parentheses is executed. This part is responsible for rendering a message when the model is loaded.
3. **`<Grid item xs={12}>`** and :
    - These are JSX elements from the Material-UI library (**`<Grid>`**). They define grid items that occupy the full width (12 columns) on extra-small screens and larger.
4. **`<Box>`**:
    - This is a Material-UI component (**`<Box>`**) used for layout and alignment within the grid items.
5. **`<CircularProgress />`**:
    - This is a Material-UI component (**`<CircularProgress>`**) representing a circular loading indicator. It's displayed when the model is still loading.
6. **`<Typography variant="body1">Loading the model...</Typography>`** and **`<Typography variant="body1">Model Loaded</Typography>`**:
    - These lines render text using the Material-UI **`<Typography>`** component.
    - The first one displays the text "Loading the model..." to inform the user that the model is currently being loaded.
    - The second one displays the text "Model Loaded" to indicate that the model has been successfully loaded.

## Similarity Matrix

```tsx
      {modelComputing ? (
        modelLoading ? (
          // Loading Model Indicator
          <Grid item xs={12}>
            <Box>
              <CircularProgress />
              <Typography variant="body1">Loading the model...</Typography>
            </Box>
          </Grid>
        ) : (
          // Model Loaded Message
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

This code controls the rendering of the similarity matrix section of the user interface based on the value of the **`showSimilarityMatrix`** state variable. If it is **`true`**, a section containing the similarity matrix is displayed. The section includes a title, "Similarity Matrix," and a canvas element for rendering the matrix. If **`false`**, nothing is rendered in this section, providing a way to show or hide the similarity matrix in the user interface.

1. **`{showSimilarityMatrix ? ... : null}`**:
    - This is a conditional statement that checks the value of the **`showSimilarityMatrix`** state variable. If **`showSimilarityMatrix`** is **`true`**, the code block inside the first set of parentheses will be executed. If it's **`false`**, **`null`** will be returned, indicating that nothing should be rendered.
2. **`<Grid item xs={12}>`**:
    - This is a JSX element from the Material-UI library (**`<Grid>`**). It defines a grid item that occupies the full width (12 columns) on extra-small screens and larger. This grid item contains the elements related to the similarity matrix.
3. **`<Paper elevation={3}>`**:
    - This is a Material-UI component (**`<Paper>`**) used for creating paper-like surfaces with an elevation effect.
4. **`<Typography variant="h3">Similarity Matrix</Typography>`**:
    - This line renders a Material-UI **`<Typography>`** component with the text "Similarity Matrix."
    - The **`variant="h3"`** prop sets the typography style to an "h3" heading, making the text larger and more prominent.
5. **`<canvas id="similarity-matrix" width={canvasSize} height={canvasSize} style={{ width: '100%', height: '100%' }}></canvas>`**:
    - This code block creates an HTML **`<canvas>`** element used to render the similarity matrix.
    - **`id="similarity-matrix"`** sets the unique ID for the canvas, which can be used for referencing and styling.
    - **`width={canvasSize}`** and **`height={canvasSize}`** set the width and height of the canvas based on the **`canvasSize`** state variable, allowing the canvas to be dynamically sized.
    - **`style={{ width: '100%', height: '100%' }}`** ensures that the canvas occupies the full width and height of its parent container.

```tsx

      {showSimilarityMatrix ? (
        // Similarity Matrix Section
        <Grid item xs={12}>
          <Paper elevation={3}>
            <Typography variant="h3">Similarity Matrix</Typography>
            <canvas id="similarity-matrix" width={canvasSize} height={canvasSize} style={{ width: '100%', height: '100%' }}></canvas>
          </Paper>
        </Grid>
      ) : null}
    </Grid>
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

