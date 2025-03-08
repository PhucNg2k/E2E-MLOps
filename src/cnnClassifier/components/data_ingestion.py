import os
import urllib.request as request
import zipfile
import gdown
from cnnClassifier import logger
from pathlib import Path
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Fetch data from the url
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            prefix_url = "https://drive.google.com/uc?export=download&id="
            file_id = dataset_url.split("/")[-2]
            gdown.download(prefix_url+file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}, size {get_size(Path(zip_download_dir))}")
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        try: 
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        except Exception as e:
            raise e    
