import pandas as pd
import numpy as np
import logging
import sys
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TimeSeriesFeatureEngineer:
    def __init__(self, input_path, output_path, lag_hours=24):
        self.input_path = input_path
        self.output_path = output_path
        self.lag_hours = lag_hours
        self.target_col = 'Global_active_power'

    def create_features(self, df):
        logger.info("Generating features...")
        
        # Lags
        for i in range(1, self.lag_hours + 1):
            df[f'lag_{i}'] = df[self.target_col].shift(i)

        # Rolling
        df['rolling_mean_6h'] = df[self.target_col].shift(1).rolling(window=6).mean()
        df['rolling_std_6h'] = df[self.target_col].shift(1).rolling(window=6).std()
        
        # Time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        df.dropna(inplace=True)
        return df

    def run(self):
        try:
            logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path, parse_dates=['datetime'], index_col='datetime')
            
            df_processed = self.create_features(df)
            
            logger.info(f"Saving features to {self.output_path}")
            df_processed.to_csv(self.output_path)
            
            logger.info(f"Feature Engineering Done. Shape: {df_processed.shape}")
            return df_processed
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_path}")
            # Don't crash main process, just return None
            return None

if __name__ == "__main__":
    engineer = TimeSeriesFeatureEngineer(
        input_path='data/processed/hourly_power_data.csv',
        output_path='data/processed/train_data.csv'
    )
    engineer.run()