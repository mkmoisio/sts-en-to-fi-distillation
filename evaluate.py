from pathlib import Path
import logging
import logging.config
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, LoggingHandler, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'
output_path = Path(os.getcwd()) / 'results'
model_path = Path(os.getcwd()) / 'models'

logging.config.fileConfig(log_path)
logger = logging.getLogger('default')




if __name__ == "__main__":
    df = pd.read_csv(data_path / 'STS-en-en-fi-fi.tsv', sep = '\t')
    #print(df.head())
    
    # Evaluators for different STS tasks
    scores = df['score']
    evaluators = []
    evaluators.append(EmbeddingSimilarityEvaluator(df['fi1'], df['fi2'], scores, name='FI-FI'))
    evaluators.append(EmbeddingSimilarityEvaluator(df['en1'], df['en2'], scores, name='EN-EN'))
    evaluators.append(EmbeddingSimilarityEvaluator(df['en1'], df['fi2'], scores, name='EN-FI'))

    # Finnish baselines, https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1
    
    word_embedding_model = models.Transformer('TurkuNLP/bert-base-finnish-cased-v1', max_seq_length=128)
    dim = word_embedding_model.get_word_embedding_dimension()
    mean_pooling = models.Pooling(dim, pooling_mode_mean_tokens=True)
    max_pooling = models.Pooling(dim, pooling_mode_max_tokens=True, pooling_mode_mean_tokens=False)
    cls_pooling = models.Pooling(dim, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    
    loaded_models = {}

    loaded_models['FinBERT-MEAN'] =  SentenceTransformer(modules=[word_embedding_model, mean_pooling])
    loaded_models['FinBERT-MAX'] = SentenceTransformer(modules=[word_embedding_model, max_pooling])
    loaded_models['FinBERT-CLS'] = SentenceTransformer(modules=[word_embedding_model, cls_pooling])

    # Trained models

    #loaded_models['Extracted'] = SentenceTransformer(model_path + '/')
    #loaded_models['Inter-lingual'] = SentenceTransformer(model_path + '/XLM-R-single-distil-3')
    #loaded_models['Cross-lingual'] = SentenceTransformer(model_path + '/XLM-R-distilled-3')
    for evaluator in evaluators:
        logger.info('Evaluating task: ' + evaluator.name)
        for model_name, model in loaded_models.items():
            logger.info('Evaluating model: ' + model_name)
            model.evaluate(evaluator, output_path)










