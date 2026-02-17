import os
import sys
import logging
from etl import ETLProcessor
from features import TimeSeriesFeatureEngineer
from trainer import MasterTrainer

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Main")
HOURLY_DATA_PATH = "data/processed/hourly_power_data.csv"
TRAIN_DATA_PATH = "data/processed/train_data.csv"

def main():
    logger.info("Starting pipeline...")
    
    if os.path.exists(HOURLY_DATA_PATH):
        logger.info("[Phase 1] ETL Cached. (hourly_power_data.csv found)")
    else:
        logger.info("\n[Phase 1] Executing ETL Pipeline...")
        ETLProcessor().run()
        
    if os.path.exists(TRAIN_DATA_PATH):
        logger.info("[Phase 2] Features Cached. (train_data.csv found)")
    else:
        logger.info("\n[Phase 2] Generating Time-Series Features...")
        engineer = TimeSeriesFeatureEngineer(
            input_path=HOURLY_DATA_PATH,
            output_path=TRAIN_DATA_PATH,
            lag_hours=24
        )
        engineer.run()

    logger.info("\n[Phase 3] Starting model benchmark...")
    logger.info("Comparing: Ridge vs LightGBM vs XGBoost vs LSTM")
    
    trainer = MasterTrainer(data_path=TRAIN_DATA_PATH)
    trainer.run_full_benchmark()
    
    logger.info("\nPipeline complete.")
    logger.info("Results: logs/benchmark_results.csv")
    logger.info("Plots: assets/")

if __name__ == "__main__":
    main()