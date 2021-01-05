from pathlib import Path
import logging
import logging.config
import pandas as pd
import os
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'
output_path = Path(os.getcwd()) / 'out'
logging.config.fileConfig(log_path)
logger = logging.getLogger('default')




if __name__ == "__main__":
    df = pd.read_csv(data_path / 'STS-en-en-fi-fi.tsv', sep = '\t')
    print(df.head())
    scores = df['score'].tolist()
    evaluators = []
    evaluators.append(EmbeddingSimilarityEvaluator(df['fi1'].tolist(), df['fi2'].tolist(), scores))
    









