# Vector Stores

Vector databases are modern data management systems designed to store and retrieve high-dimensional vectors 
representing unstructured data.

## Introduction

### What is unstructured data
Unstructured data refers to information that doesn't have a predefined data model or organized format. Unlike 
structured data, which fits neatly into tables with rows and columns, unstructured data lacks a consistent structure, 
making it more challenging to store, process, and analyze using traditional databases and methods.
Unstructured data can take various forms, including:

1. Textual Data: This includes documents, emails, social media posts, and other text-based content that doesn't adhere 
to a strict structure.
2. Images and Videos: Visual content such as images and videos don't have a tabular structure and are considered 
unstructured data.
3. Audio: Audio recordings, like spoken conversations or music tracks, are also unstructured in nature.
4. Sensor Data: Data collected from various sensors, such as those in IoT devices, might not follow a predefined format.
5. Web Pages: Web pages often contain a mix of text, images, and other media, resulting in unstructured content.
6. Social Media Posts: Posts on platforms like Twitter, Facebook, and Instagram often contain text, images, and 
hashtags, with no uniform structure.
7. Free-Form Surveys: Open-ended survey responses where respondents can provide any kind of text-based answer are 
considered unstructured data.

Due to its diverse and variable nature, unstructured data poses challenges for traditional databases that are designed 
to handle structured data efficiently. Vector stores provide a solution to unstructured data by leveraging vector 
space relations.

### How does this work
Taking a set of textual data in the form of a set of PDF documents. 
Each document is broken down into a set of chucks of textual meaing. 
These chunks are presented as input to a Machine Learning model such as _Word2vec_ that 
transforms the chuck of textual data into a high dimensional vector. This vector becomes what is known as an **Embedding**.
This forms a numerical representation of the textual data.

**Embedding Def**: An embedding is a vector representation that encodes the underlying meaning behind unstructured data.

Take for example the below vector space generated from a set of words:

<p align="center">
    <img src="resources/word_embedding.png" width="470" height="425" alt="Word Embedding">
</p>

#### Embedding Arithmetic
Within a vector space, the embeddings are subject to the mathematics underlying laws of linear algebra.
The following linear algebra actions can be applied:

1. Vector Addition and Subtraction: Adding or subtracting vectors can highlight relationships between them. For example, in the context of word embeddings, the operation "king - man + woman" might result in a vector close to the "queen" vector, demonstrating the ability to capture analogies.
2. Cosine Similarity: Cosine similarity measures the cosine of the angle between two vectors. It quantifies the similarity between vectors in terms of their direction, not their magnitude. This is often used to find similarity between embeddings, like measuring the similarity between two text documents.
3. Nearest Neighbor Search: Given an embedding vector, a nearest neighbor search identifies other vectors in the dataset that are closest to the given vector in terms of Euclidean distance, cosine similarity, or other distance metrics. This is useful for recommendation systems and content retrieval.
4. Clustering: Grouping similar vectors together based on their proximity in the vector space is known as clustering. K-means clustering and hierarchical clustering are common techniques used with embeddings to group similar data points.
5. Principal Component Analysis (PCA) and Dimensionality Reduction: PCA is a technique to reduce the dimensionality of the data while preserving its most important variations. It can be applied to embeddings to simplify visualization and analysis while retaining essential information.
6. Arithmetic Operations on Image Embeddings: In computer vision, embeddings can represent images. You can perform arithmetic operations on these embeddings to achieve interesting effects, like image style transfer or morphing.
7. Time Series Analysis: Embeddings of time series data can be used to detect patterns and anomalies, aiding in forecasting and anomaly detection tasks.
8. Text Similarity and Sentiment Analysis: By measuring the similarity between text embeddings, you can gauge the semantic similarity between different text documents. This is used in sentiment analysis, text classification, and recommendation systems.
9. Interpolation and Extrapolation: By linearly interpolating between two embeddings, you can generate new embeddings representing intermediate states. Extrapolation extends this to create embeddings representing points beyond the original data distribution.
10. Attention Mechanisms: In natural language processing, attention mechanisms leverage vector actions to assign different weights to different parts of a sequence, allowing models to focus on specific elements during tasks like translation or text summarization.
11. Correlation: Correlation between vectors within the space can reveal underlying semantics and relations within the vector space. The example below visualizes this concept

<p align="center">
    <img src="resources/vector_correlation.png" width="470" height="425" alt="Correlation in Vector Space">
    Source: The Evolution of Milvus: A Cloud-Native Vector Database - Frank Liu, Zilliz
</p>

<p align="center">
    <img src="resources/image_similarity.png" width="470" height="425" alt="Image similarity search">
    Source: The Evolution of Milvus: A Cloud-Native Vector Database - Frank Liu, Zilliz
</p>


### Process Summary
1. Text Embedding: Before storage, the textual data is transformed into numerical representations called text embeddings. This conversion involves techniques like Word2Vec, GloVe, or more advanced language models like BERT or GPT, which map words or phrases to dense vectors in a semantic space.
2. Vector Representation: Each text embedding becomes a vector in a high-dimensional space, where the dimensions correspond to different features or aspects of the text's meaning. Words or phrases with similar meanings are positioned closer to each other in this vector space.
3. Indexing: Vector databases employ specialized indexing structures optimized for vector data. These indexes store information about the vectors' positions, allowing for efficient retrieval of similar vectors or nearest neighbors.
4. Similarity Search: When querying the database with a specific text, the query text is transformed into a vector using the same embedding technique. The database then performs a similarity search, identifying the nearest vectors in the high-dimensional space to the query vector. These nearest vectors represent texts with similar semantic content.
5. Ranking and Retrieval: The retrieved vectors are ranked based on their similarity to the query vector. The top-ranked vectors represent the most similar texts, and these can be returned as search results.
6. Applications: Vector databases storing textual data find applications in various fields. They power search engines, recommendation systems, content similarity detection, sentiment analysis, and more, where understanding the semantic relationships between texts is crucial.

## Potential Vector Storage Solutions


## Sources
- [The Evolution of Milvus: A Cloud-Native Vector Database - Frank Liu, Zilliz](https://youtu.be/4yQjsY5iD9Q?si=u58fMUuCaL1oDoN4)

