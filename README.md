# sts-en-to-fi-distillation

# WIP

## Overview

Semantic textual similarity (STS) is a metric measuring similarity of meanings of natural language text pieces. For instance, sentences "There is a cook preparing food" and "A cook is making food" (from SemEval-2017 set) have essentially equal meaning and, therefore, should have very high STS score. STS is quite well studied area in NLP. SemEval, an evaluation semantic analysis systems, kicked off in 2007 and has been held annually since 2012. However, despite recent progress, STS is nowhere close to being a solved problem. Natural language is diverse, ambiguous, and highly context-dependent. There are thousands of languages, but NLP research has mainly focused on a small subset languages, that is, few European languages with the largest speaker base and Chinese.

Sentence BERT is an extension proposed by Reimers et al. to BERT-type family of models. Its idea is to fine-tune a model to produce semantically meaningful sentence embeddings that one can compare with simple vector similarity measures (e.g. manhattan distance or cosine similarity). Further, Reimers et al. have described a method, multilingual distillation, to extend a monolingual SentenceBERT to a cross-lingual model.

This work explores capacity of distillation-like techniques in different settings. Common to all settings is that given a known-good monolingual teacher model we aim to transfer its desired properties to another model and language. In our case the source and target languages are English and Finnish, respectively.





## Models

## Data

| Set   |      Type      |  Size |
|----------|:-------------:|------:|
| Training, cross-lingual |   Parallel bilingual | 250,00 |
| Training, inter-lingual |    Unilingual   |   250,00 |
| Training, extraction  | Unilingual |   250,00 |
| Validation | Parallel bilingual| 1,500 |
| Test  | Annotated bilingual | 250 |

## Training

## Evaluation Metrics
