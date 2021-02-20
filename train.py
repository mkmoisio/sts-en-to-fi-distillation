from sentence_transformers import SentenceTransformer, models, evaluation, losses
from torch.utils.data import DataLoader
from torch import nn, Tensor
from sentence_transformers.datasets import ParallelSentencesDataset
import configparser
import os, csv, torch, datetime
from typing import Dict
from utils.utils import download_dataset
import pandas as pd
from pathlib import Path
import logging
import logging.config

log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'
output_path = Path(os.getcwd()) / 'models'
logging.config.fileConfig(log_path)
logger = logging.getLogger('default')



def download_datasets(datasets: Dict):
    for key in datasets:
        file_name = datasets[key]
        if os.path.isfile(data_path / file_name):
            logger.info('File "' + file_name + '" exists, continuing')
        else:
            download_dataset(file_name)




if __name__ == "__main__":

    logger.info('Loading config...')
    config = configparser.ConfigParser()
    config.read(os.getcwd() + '/config/distil.conf')
    model_names = config['MODELS']
    model_conf = config['MODELSETUP']
    datasets = config['DATASETS']
    train_conf = config['TRAIN']
    eval_conf = config['EVALUATION']

    optimizer_params = config['OPTIMIZER']

    optimizer_params = {
        'lr': optimizer_params.getfloat('lr'),
        'eps': optimizer_params.getfloat('eps'),
        'correct_bias': optimizer_params.getboolean('correct_bias')
    }

    logger.info('Downloading required datasets..')
    download_datasets(datasets)

   # print(config.sections())
    
    logger.info('Loading teacher model...')
    teacher = SentenceTransformer(model_names['Teacher'])
   

    logger.info('Loading student model...')
    embedding_model = models.Transformer(model_names['Student'], max_seq_length=model_conf.getint('MaxSeqLen'))
    pooling = models.Pooling(embedding_model.get_word_embedding_dimension())
    student = SentenceTransformer(modules=[embedding_model, pooling])    

    logger.info('Loading training set...')
    train_data = ParallelSentencesDataset(student_model=student, teacher_model=teacher)
    # ParallelSentencesDataset can't handle PosixPaths, therefore cast to string
    train_set_path = str(data_path) + '/' + datasets['TrainSet']
    train_data.load_data(train_set_path , max_sentences=None, max_sentence_length=train_conf.getint('MaxSentLen'))
    train_dataloader = DataLoader(train_data, batch_size=train_conf.getint('BatchSize'))

    #train_loss = CosineSimilarityLoss(model=student_model)
    train_loss  = losses.MSELoss(model=student)

    logging.info('Assembling evaluator')
    df = pd.read_csv(data_path / datasets['DevSet'], sep = '\t', header = None, quoting=3)
    dev_mse_evaluator = evaluation.MSEEvaluator(df.iloc[:, 0], df.iloc[:, 1], name='Dev-MSE-evaluator', teacher_model=teacher, batch_size=eval_conf.getint('BatchSize'))

    logging.info('Fitting..')
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=2))).strftime("%Y-%m-%d-%H:%M:%S")
    output_path = output_path / dt

    student.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_mse_evaluator,
          epochs=train_conf.getint('Epochs'),
          steps_per_epoch= train_conf.getint('Steps'),
          scheduler = config['SCHEDULER']['Scheduler'],
          warmup_steps=train_conf.getint('WarmUp'),
          evaluation_steps=eval_conf.getint('EvalSteps'),
          output_path=output_path,
          save_best_model=eval_conf.getboolean('SaveBest'),
          optimizer_params= optimizer_params
          )





