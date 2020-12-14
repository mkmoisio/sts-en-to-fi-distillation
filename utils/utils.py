import os
import sys
import requests
from pathlib import Path
import logging
import logging.config

log_path = Path(os.getcwd()) / 'config/logging.conf'
data_path = Path(os.getcwd()) / 'data'
dl_url = 'https://nlpdistillation.blob.core.windows.net/nlpdistillation/'


logging.config.fileConfig(log_path)
logger = logging.getLogger('default')




def download_dataset(file_name):

    logger.info('Downloading file "' + file_name +
                '" from ' + dl_url + file_name)
    response = requests.get(dl_url + file_name)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        logger.info('Saving file ' + file_name + ' to ' + str(data_path))
        with open(data_path / file_name, mode='w') as f:
            f.write(response.text)

    else:
        logger.error('Failed to download file ' + file_name)


if __name__ == "__main__":
    download_data(sys.argv[1])
