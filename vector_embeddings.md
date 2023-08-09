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

## Sources
- [The Evolution of Milvus: A Cloud-Native Vector Database - Frank Liu, Zilliz](https://youtu.be/4yQjsY5iD9Q?si=u58fMUuCaL1oDoN4)

