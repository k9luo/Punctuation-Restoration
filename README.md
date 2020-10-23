Punctuation Restoration
====================================================================
![](https://img.shields.io/badge/linux-ubuntu-red.svg)

![](https://img.shields.io/badge/cuda-10.0.130-green.svg)
![](https://img.shields.io/badge/python-3.7.6-green.svg)

![](https://img.shields.io/badge/tensorflow-1.14.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.19.1-blue.svg)
![](https://img.shields.io/badge/ujson-4.0.1-blue.svg)
![](https://img.shields.io/badge/jupyter-1.0.0-blue.svg)
![](https://img.shields.io/badge/pandas-1.1.3-blue.svg)
![](https://img.shields.io/badge/tqdm-4.50.2-blue.svg)

## Requirements

Imagine that you are building a software for transcribing speech to text. The speech transcription part works perfectly, but cannot transcribe punctuations. The task is to train a predictive model to ingest a sequence of text and add punctuation (period, comma or question mark) in the appropriate locations. This task is important for all downstream data processing jobs.

**Example input:**
 
```this is a string of text with no punctuation this is a new sentence```
 
**Example output:**
 
```this is a string of text with no punctuation <period> this is a new sentence <period>```

## Solution

My solution is largely based on [Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF).

The architecture is defined as follows:
1. Obtain words embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/).
2. The word embeddings are then processed by densely connected [Bi-LSTM](https://arxiv.org/pdf/1303.5778.pdf) layers.
3. These Bi-LSTM layers are followed by a RNN with an attention mechanism and [conditional random field (CRF)](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) log likelihood loss.

The experiments are performed on the IWSLT dataset which consists of TED Talks transcript.

The detailed analysis can be found in this [notebook](https://github.com/k9luo/Punctuation-Restoration/blob/main/main.ipynb).

## Setup and Installation

First step, clone the repo:

```https://github.com/k9luo/Punctuation-Restoration.git```

Second step, you can download pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings, create a new conda virutal environment and add the new virutal environment to Jupyter Notebook with `setup.sh`. Or you can manually do these steps yourself. Note that the running `setup.sh` will install the GPU version of TensorFlow:

```sh setup.sh```

Third step, activate the virtual environment:

```conda activate restore_punct```

## Training and Inference

Please run `python main.py`.
