import os
import sys
import logging
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from diagnostics import ResearchDiagnostics
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ASSETS_DIR = "assets"
LOGS_DIR = "logs"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{LOGS_DIR}/full_benchmark.log")
    ]
)
logger = logging.getLogger("Benchmark")
class PowerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(PowerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class MasterTrainer:
    def __init__(self, data_path, target_col='Global_active_power'):
        self.data_path = data_path
        self.target_col = target_col
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.diagnostics = ResearchDiagnostics(save_dir=ASSETS_DIR)
        self.feature_names = [] # To store column names

    def load_data(self):
        df = pd.read_csv(self.data_path, index_col='datetime', parse_dates=True)
        if df.isnull().values.any():
            df = df.ffill().fillna(0)
        self.feature_names = df.drop(columns=[self.target_col]).columns.tolist()
        return df

    def get_splits(self, df):
        X = df.drop(columns=[self.target_col]).values.astype(np.float32)
        y = df[self.target_col].values.astype(np.float32).reshape(-1, 1)
        
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_raw, y_test_raw = y[train_idx], y[test_idx]
            
            X_train = self.scaler_X.fit_transform(X_train_raw)
            X_test = self.scaler_X.transform(X_test_raw)
            y_train = self.scaler_y.fit_transform(y_train_raw)
            
            yield X_train, y_train, X_test, y_test_raw

    
    def train_ridge(self, X_train, y_train, X_test, y_test, params):
        try:
            model = Ridge(alpha=params['alpha'], random_state=42)
            model.fit(X_train, y_train.ravel())
            preds_scaled = model.predict(X_test)
            preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds_real))
            return rmse, model, preds_real
        except Exception:
            return float('inf'), None, None

    def train_lightgbm(self, X_train, y_train, X_test, y_test, params):
        try:
            X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
            X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
            
            model = lgb.LGBMRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['lr'],
                num_leaves=params['num_leaves'],
                n_jobs=1,
                random_state=42,
                verbosity=-1
            )
            model.fit(X_train_df, y_train.ravel())
            
            preds_scaled = model.predict(X_test_df)
            preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds_real))
            return rmse, model, preds_real
        except Exception as e:
            return float('inf'), None, None

    def train_xgboost(self, X_train, y_train, X_test, y_test, params):
        try:
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['lr'],
                n_jobs=1,
                random_state=42
            )
            model.fit(X_train, y_train.ravel())
            preds_scaled = model.predict(X_test)
            preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds_real))
            return rmse, model, preds_real
        except Exception:
            return float('inf'), None, None

    def train_lstm(self, X_train, y_train, X_test, y_test, params):
        try:
            dropout_val = 0.0 if params['num_layers'] == 1 else params['dropout']
            
            X_tr_t = torch.tensor(X_train).unsqueeze(1)
            y_tr_t = torch.tensor(y_train).float()
            X_te_t = torch.tensor(X_test).unsqueeze(1)
            
            model = PowerLSTM(X_train.shape[1], params['hidden_dim'], params['num_layers'], dropout_val)
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(20): 
                optimizer.zero_grad()
                out = model(X_tr_t)
                loss = criterion(out, y_tr_t)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                preds_scaled = model(X_te_t).numpy()
                
            preds_real = self.scaler_y.inverse_transform(preds_scaled).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, preds_real))
            return rmse, model, preds_real
        except Exception:
            return float('inf'), None, None

    def run_full_benchmark(self):
        df = self.load_data()
        
        self.diagnostics.check_stationarity(df[self.target_col], "Target Variable")

        results = []
        
        ridge_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
        
        lgbm_grid = {
            'n_estimators': [100, 300, 500],
            'lr': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127] 
        }
        
        xgb_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 6, 9],
            'lr': [0.01, 0.05, 0.1]
        }
        
        lstm_grid = {
            'hidden_dim': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout': [0.0, 0.2, 0.5],
            'lr': [0.01, 0.001]
        }

        def run_grid(name, grid, train_func):
            logger.info(f"--- Tuning {name} ---")
            keys, vals = zip(*grid.items())
            combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
            
            logger.info(f"Testing {len(combos)} configurations for {name}...")
            
            best_model_rmse = float('inf')
            best_model_preds = None
            last_y = None
            
            for params in tqdm(combos, desc=name):
                fold_scores = []
                for fold_idx, (X_tr, y_tr, X_te, y_te) in enumerate(self.get_splits(df)):
                    rmse, _, preds = train_func(X_tr, y_tr, X_te, y_te, params)
                    fold_scores.append(rmse)
                    
                    if fold_idx == 2: 
                        last_preds = preds
                        last_y = y_te.flatten()

                avg_rmse = np.mean(fold_scores)
                results.append({'Model': name, 'Params': str(params), 'RMSE': avg_rmse})
                
                if avg_rmse < best_model_rmse:
                    best_model_rmse = avg_rmse
                    best_model_preds = last_preds
            
            if best_model_preds is not None:
                logger.info(f"Best {name} RMSE: {best_model_rmse:.4f}")
                self.diagnostics.analyze_residuals(last_y, best_model_preds, name)
            else:
                logger.error(f"All configurations for {name} failed. Skipping diagnostics.")

        run_grid("Ridge", ridge_grid, self.train_ridge)
        run_grid("LightGBM", lgbm_grid, self.train_lightgbm)
        run_grid("XGBoost", xgb_grid, self.train_xgboost)
        run_grid("LSTM", lstm_grid, self.train_lstm)

        df_res = pd.DataFrame(results).sort_values(by='RMSE')
        df_res.to_csv(f"{LOGS_DIR}/benchmark_results.csv", index=False)
        self.diagnostics.generate_report(df_res)
        
        print("\nBenchmark results:")
        print("-"*50)
        print(df_res.head(10))

if __name__ == "__main__":
    trainer = MasterTrainer(data_path='data/processed/train_data.csv')
    trainer.run_full_benchmark()