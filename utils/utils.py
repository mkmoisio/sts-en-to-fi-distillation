import os, sys
import requests
from pathlib import Path
import logging
import logging.config

path = Path(os.getcwd()) / 'config/logging.conf'
logging.config.fileConfig(path)
logger = logging.getLogger('default')


DL_URL = 'https://nlpdistillation.blob.core.windows.net/nlpdistillation/'
OUTPUT_PATH = Path(os.getcwd()) / 'data'


def download_data(file_name):
    logger.info('Downloading file "' + file_name + '" from ' + DL_URL + file_name)
    response = requests.get(DL_URL + file_name)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        logger.info('Saving file ' + file_name + 'from' + OUTPUT_PATH)
        with open(OUTPUT_PATH / file_name, mode = 'w') as f:
            f.write(response.text)

    else:
        logger.error('Failed to download file ' + file_name)

download_data('train-set.tsv')


