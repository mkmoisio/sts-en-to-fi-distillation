from pathlib import Path
import logging
import logging.config
import pandas as pd
import numpy as np
import os
import configparser
import zipfile
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, LoggingHandler, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import simplejson as json
import gdown
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'
output_path = Path(os.getcwd()) / 'results'
model_path = os.getcwd() + '/models/'

logging.config.fileConfig(log_path)
logger = logging.getLogger('default')


def dl_unzip(frm, name):
    zfpath = str(model_path) + name + '.zip'
    if not os.path.exists(zfpath):
        gdown.download(frm, model_path + name + '.zip', quiet=False)
    with zipfile.ZipFile(zfpath, 'r') as zf:
        zf.extractall(path=model_path)


if __name__ == "__main__":
    df = pd.read_csv(data_path / 'STS-en-en-fi-fi.tsv', sep = '\t')

    config = configparser.ConfigParser()
    config.read(os.getcwd() + '/config/eval.conf')
    model_urls = config['MODELS']

    # Evaluators for different STS tasks
    scores = df['score']
    evaluators = []
    
    evaluators.append(EmbeddingSimilarityEvaluator(df['fi1'], df['fi2'], scores, name='FI-FI', main_similarity = SimilarityFunction.COSINE))
    evaluators.append(EmbeddingSimilarityEvaluator(df['en1'], df['en2'], scores, name='EN-EN', main_similarity = SimilarityFunction.COSINE))
    evaluators.append(EmbeddingSimilarityEvaluator(df['en1'], df['fi2'], scores, name='EN-FI', main_similarity = SimilarityFunction.COSINE))

    
    
    loaded_models = {}

    # Trained student models
    for name in model_urls:
        dl_unzip(model_urls[name], name)
        loaded_models[name] = SentenceTransformer(model_path + name)
    # Finnish baselines, https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1
    
    word_embedding_model = models.Transformer('TurkuNLP/bert-base-finnish-cased-v1', max_seq_length=128)
    dim = word_embedding_model.get_word_embedding_dimension()
    mean_pooling = models.Pooling(dim, pooling_mode_mean_tokens=True)
    max_pooling = models.Pooling(dim, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=False)
    cls_pooling = models.Pooling(dim, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    
    

    loaded_models['FinBERT-MEAN'] =  SentenceTransformer(modules=[word_embedding_model, mean_pooling])
    loaded_models['FinBERT-MAX'] = SentenceTransformer(modules=[word_embedding_model, max_pooling])
    loaded_models['FinBERT-CLS'] = SentenceTransformer(modules=[word_embedding_model, cls_pooling])
    loaded_models[name] = SentenceTransformer(model_path + name)

    results = {}
    for evaluator in evaluators:
        logger.info('Evaluating task: ' + evaluator.name)
        results[evaluator.name] = {}
        for model_name, model in loaded_models.items():
            logger.info('Evaluating model: ' + model_name)
            spearman_cosine = model.evaluate(evaluator)
            results[evaluator.name][model_name] = spearman_cosine 
    with open(output_path / 'results.json', 'w') as fp:        
        json.dump(results, fp)


    colors = mcolors.TABLEAU_COLORS
    for item in results.items():
        task = item[0]
        labels = item[1].keys()
        scores = item[1].values()
        fig, ax1 = plt.subplots(figsize=(8, 6))

        index = np.arange(len(scores))
        plt.xticks(index, labels)
        plt.ylim([0.0, 1.0])
        plt.ylabel('Spearman\'s ρ,' + task)
        plt.bar(index, scores, color = colors)
        plt.savefig(str(output_path) +'/' + task + '.png')







