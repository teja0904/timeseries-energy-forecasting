import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Diagnostics")

class ResearchDiagnostics:
    def __init__(self, save_dir="assets"):
        self.save_dir = save_dir

    def check_stationarity(self, timeseries, title="Raw Data"):
        logger.info(f"--- Running ADF Test on {title} ---")
        result = adfuller(timeseries.dropna())
        p_value = result[1]
        
        logger.info(f"ADF Statistic: {result[0]:.4f}")
        logger.info(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            logger.info(">> REJECT H0: Data is Stationary (Good for Modeling)")
            return True
        else:
            logger.warning(">> FAIL TO REJECT H0: Data is Non-Stationary (Needs Differencing)")
            return False

    def analyze_residuals(self, y_true, y_pred, model_name):
        residuals = y_true - y_pred
        
        logger.info(f"--- Analyzing Residuals for {model_name} ---")
        
        if len(residuals) > 5000:
            sample_resid = np.random.choice(residuals, 5000, replace=False)
            stat, p = stats.shapiro(sample_resid)
        else:
            stat, p = stats.shapiro(residuals)
            
        logger.info(f"Shapiro-Wilk Test: p={p:.6f}")
        if p < 0.05:
            logger.warning(">> Residuals are NOT Normally Distributed (Model missed some patterns)")
        else:
            logger.info(">> Residuals look Gaussian (Good fit)")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Research Diagnostics: {model_name}", fontsize=16)

        stats.probplot(residuals, dist="norm", plot=axes[0,0])
        axes[0,0].set_title("Q-Q Plot (Normality Check)")

        sns.scatterplot(x=y_pred, y=residuals, ax=axes[0,1], alpha=0.1, color='teal')
        axes[0,1].axhline(0, color='red', linestyle='--')
        axes[0,1].set_title("Residuals vs. Predicted (Homoskedasticity Check)")
        axes[0,1].set_xlabel("Predicted Power")
        axes[0,1].set_ylabel("Residual Error")

        plot_acf(residuals, ax=axes[1,0], lags=40, title="Residual Autocorrelation (ACF)")
        sns.histplot(residuals, kde=True, ax=axes[1,1], color='purple', bins=50)
        axes[1,1].set_title(f"Error Distribution (Mean: {np.mean(residuals):.4f})")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/diagnostics_{model_name}.png")
        plt.close()

    def generate_report(self, results_df):
        logger.info("Generating LaTeX Comparison Table...")
        latex_code = results_df.to_latex(index=False, float_format="%.4f")
        with open(f"{self.save_dir}/final_table.tex", "w") as f:
            f.write(latex_code)
        logger.info(f"LaTeX table saved to {self.save_dir}/final_table.tex")