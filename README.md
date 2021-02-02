# sts-en-to-fi-distillation

# WIP

## Overview

Semantic textual similarity (STS) is a metric measuring similarity of meanings of natural language text pieces. For instance, sentences "There is a cook preparing food" and "A cook is making food" (from SemEval-2017 set) have essentially equal meaning and, therefore, should have very high STS score. STS is quite long studied area in the field NLP. However, despite recent progress, STS is nowhere close being solved and majority of recent NLP research has focused on a small set of largest Indo-European languages and Chinese, and, languages with smaller speaker population, such as Finnish, often lack annotated data required to train complex models.

This work experiments  using knowledge distillation to transfer STS properties learnt from English into a model pre-trained on Finnish while bypassing the lack of annotated Finnish data. Further, we experiment distillation with different types of data, English-Finnish bilingual, English monolingual and random pseudo samples, to observe which properties of training data are really necessary.

This projct builds on the work by the following:
* Reimers et al.
  - [Sentence-BERT](https://arxiv.org/abs/1908.10084) ([repository](https://github.com/UKPLab/sentence-transformers))
  -  [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813)
* Krishna et al.
  - [Model Extraction](https://arxiv.org/abs/1910.12366)



## Models

## Data

| Set   |      Type      |  Size |
|----------|:-------------:|------:|
| Training, cross-lingual |   Parallel bilingual | 216,104 |
| Training, inter-lingual |    Unilingual   |   208,054 |
| Training, extraction  | Sampled |   200,000 |
| Development | Parallel bilingual| 1,141 |
| Test  | Annotated bilingual | 250 |

## Training

## Evaluation Metrics
