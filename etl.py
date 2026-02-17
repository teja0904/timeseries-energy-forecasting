import os
import sys
import logging
import zipfile
import requests
import io
import pandas as pd
import numpy as np
from tqdm import tqdm

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/etl_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ETLProcessor:
    def __init__(self):
        self.data_url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
        self.raw_dir = os.path.join("data", "raw")
        self.processed_dir = os.path.join("data", "processed")
        self.raw_filename = "household_power_consumption.txt"
        self.processed_filename = "hourly_power_data.csv"
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_data(self):
        extract_path = os.path.join(self.raw_dir, self.raw_filename)
        
        if os.path.exists(extract_path):
            logger.info("Raw data found locally. Skipping download.")
            return

        logger.info(f"Downloading data from {self.data_url}...")
        
        try:
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with io.BytesIO() as buffer:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        buffer.write(chunk)
                        pbar.update(len(chunk))
                
                logger.info("Download complete. Extracting...")
                with zipfile.ZipFile(buffer) as z:
                    z.extractall(self.raw_dir)
                    
            logger.info(f"Data extracted to {self.raw_dir}")
            
        except Exception as e:
            logger.error(f"Failed to download or extract data: {e}")
            raise e

    def load_and_clean(self):
        file_path = os.path.join(self.raw_dir, self.raw_filename)
        logger.info("Loading raw text file into DataFrame...")
        
        df = pd.read_csv(file_path, sep=';', na_values=['?'], low_memory=False)
        logger.info(f"Raw shape: {df.shape}")
        
        df.dropna(inplace=True)

        logger.info("Parsing datetime index...")
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df.set_index('datetime', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        df = df.astype('float32')
        return df

    def resample_data(self, df):
        logger.info("Resampling data to Hourly frequency...")
        df_hourly = df.resample('h').mean()
        
        if df_hourly.isnull().sum().sum() > 0:
            df_hourly = df_hourly.ffill()
            
        return df_hourly

    def run(self):
        self.download_data()
        df_raw = self.load_and_clean()
        df_hourly = self.resample_data(df_raw)
        
        save_path = os.path.join(self.processed_dir, self.processed_filename)
        df_hourly.to_csv(save_path)
        logger.info(f"ETL Pipeline complete. Processed data saved to {save_path}")
        return df_hourly

if __name__ == "__main__":
    processor = ETLProcessor()
    processor.run()