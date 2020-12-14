from sentence_transformers import SentenceTransformer, models, evaluation, losses
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch
from sentence_transformers.datasets import ParallelSentencesDataset
import configparser
import os
from typing import Dict
from utils.utils import download_dataset


log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'

logging.config.fileConfig(path)
logger = logging.getLogger('default')



def download_datasets(datasets: Dict):
    logger.info('Downloading required datasets..')
    for key in datasets:
        if os.path.isfile(data_path / file_name):
            logger.info('File "' + file_name + '" exists, continuing')
        else:
            download_dataset(datasets[key])



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.getcwd() + '/config/distil.conf')
    
    download_datasets(config['DATASETS'])

    print(config.sections())








