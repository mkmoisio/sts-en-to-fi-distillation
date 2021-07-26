## Overview

Semantic textual similarity (STS) is a metric measuring similarity of meanings of natural language text pieces. Majority of recent NLP research has focused on a small set of largest Indo-European languages and Chinese, and, languages with smaller speaker population, Finnish for instance, often lack annotated data required to train complex models.

This work experiments using knowledge distillation to transfer STS properties learnt from English into a model pre-trained on Finnish while bypassing the lack of annotated Finnish data. Further, we experiment distillation with different types of data, English-Finnish bilingual, English monolingual and random pseudo samples, to observe which properties of training data are really necessary.



We briefly follow the knowledge distillation approach described by Reimers at al. We exploit a good-known English trained Sentence-BERT  fine-tuned for STS as a teacher f<sub>teacher</sub> to train selected student models f<sub>student</sub> so that f<sub>teacher</sub>(s) ≈ f<sub>student</sub>(s) and f<sub>student</sub>(s) ≈ f<sub>student</sub> (t) where s is a sentence in the source language and t is a sentence in target language.

## Models

| Model   |      Role      |  Origin |
|----------|:-------------:|------:|
| stsb-distilbert-base| Teacher + Baseline | [sbert.net](https://www.sbert.net/docs/pretrained_models.html)|
| bert-base-finnish-cased-v1 (FinBERT) |    Baseline   |  [TurkuNLP](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1)|
| xlm-roberta-base  | Student + Baseline | [🤗](https://huggingface.co/transformers/model_doc/xlmroberta.html) |



## Data

| Set   |      Type      |  Size |
|----------|:-------------:|------:|
| [Training, cross-lingual](https://drive.google.com/file/d/1klA2zLcFvrLQE7LHXWpMIf0O_Rad-eDT/view?usp=sharing) |   Parallel bilingual | 216,104 |
| [Training, inter-lingual](https://drive.google.com/file/d/1kZoVVZdzG3pd0CjgJM7Eo1YoiXwg9qFf/view?usp=sharing) |    Unilingual   |   208,054 |
| [Training, extraction](https://drive.google.com/file/d/1kZoVVZdzG3pd0CjgJM7Eo1YoiXwg9qFf/view?usp=sharing)  | Sampled |   200,000 |
| [Development](https://drive.google.com/file/d/1i1BY0CyFsuAXtbw8uoQvzEZ53qQwv-nc/view?usp=sharing) | Parallel bilingual| 1,141 |
| [Test](https://github.com/mkmoisio/sts-en-to-fi-distillation/blob/master/data/STS-en-en-fi-fi.tsv) | Annotated bilingual | 250 |

## Training

## Evaluation


## Ref

This work mostly builds on the following:
* Reimers et al.
  - [Sentence-BERT](https://arxiv.org/abs/1908.10084) ([repository](https://github.com/UKPLab/sentence-transformers))
  -  [Multilingual Knowledge Distillation](https://arxiv.org/abs/2004.09813)
* Krishna et al.
  - [Model Extraction](https://arxiv.org/abs/1910.12366)
*  Virtanen et al.
  - [FinBERT](https://arxiv.org/abs/1912.07076) ([repository](https://github.com/TurkuNLP/FinBERT))
* Tiedemann
  - [Parallel data](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf) ([OPUS](https://opus.nlpl.eu))
* Cer et al.
  - [Original test data](https://www.aclweb.org/anthology/S17-2001/)
