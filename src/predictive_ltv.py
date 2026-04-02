import sys
import inspect
import numpy as np

# --- DER ULTIMATIVE MONKEY PATCH (Muss GANZ oben stehen) ---
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# NumPy 2.0 Patch 1: Das gelöschte 'msort'
if not hasattr(np, 'msort'):
    np.msort = np.sort

# NumPy 2.0 Patch 2: Der "copy=False" Fehler in autograd
_original_array = np.array
def _patched_array(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
        return np.asarray(*args, **kwargs)
    return _original_array(*args, **kwargs)
np.array = _patched_array
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class PredictiveLTVModeler:
    """
    Nutzt das 'lifetimes' Framework für stabile LTV-Vorhersagen.
    """
    def __init__(self, data_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.observation_period_end = '2023-12-31'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_prep_rfm_data(self):
        print("Lade Transaktionsdaten...")
        self.transactions = pd.read_csv(f"{self.data_dir}/transactions.csv", parse_dates=['transaction_date'])
        self.customers = pd.read_csv(f"{self.data_dir}/customers.csv")
        
        print("Erstelle RFM-Summary (Das dauert einen kurzen Moment)...")
        self.rfm = summary_data_from_transaction_data(
            self.transactions, 
            customer_id_col='customer_id', 
            datetime_col='transaction_date', 
            monetary_value_col='revenue', 
            observation_period_end=self.observation_period_end
        )
        
        # Merge mit Kanälen
        self.rfm = self.rfm.reset_index().merge(
            self.customers[['customer_id', 'acquisition_channel']], 
            on='customer_id', 
            how='inner'
        ).set_index('customer_id')
        
        # Für Gamma-Gamma brauchen wir Kunden mit mind. 1 Repeat-Purchase
        self.rfm_repeat = self.rfm[(self.rfm['frequency'] > 0) & (self.rfm['monetary_value'] > 0)]
        print(f"Kunden gesamt: {len(self.rfm)}. Davon Wiederkäufer: {len(self.rfm_repeat)}")

    def fit_models(self):
        print("Training BG/NBD Modell (Kaufhäufigkeit)...")
        self.bgf = BetaGeoFitter(penalizer_coef=0.01)
        self.bgf.fit(self.rfm['frequency'], self.rfm['recency'], self.rfm['T'])
        
        # Vorhersage für 6 Monate (180 Tage)
        self.rfm['predicted_purchases_6m'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
            180, self.rfm['frequency'], self.rfm['recency'], self.rfm['T']
        )
        
        # P-Alive Berechnung
        self.rfm['p_alive'] = self.bgf.conditional_probability_alive(
            self.rfm['frequency'], self.rfm['recency'], self.rfm['T']
        )

        print("Training Gamma-Gamma Modell (Geldwert)...")
        self.ggf = GammaGammaFitter(penalizer_coef=0.01)
        self.ggf.fit(self.rfm_repeat['frequency'], self.rfm_repeat['monetary_value'])
        
        # Vorhergesagter AOV
        self.rfm['predicted_aov'] = self.ggf.conditional_expected_average_profit(
            self.rfm['frequency'], self.rfm['monetary_value']
        )
        
        # Fallback für Einmalkäufer
        mean_aov = self.rfm['monetary_value'].mean()
        self.rfm['predicted_aov'] = self.rfm['predicted_aov'].fillna(mean_aov)

    def finalize_predictions(self):
        print("Berechne finalen 6-Monats-LTV...")
        # LTV = Erwartete Käufe * Erwarteter Wert
        self.rfm['predicted_6m_ltv'] = self.rfm['predicted_purchases_6m'] * self.rfm['predicted_aov']
        
        # Ergebnis speichern
        output_path = f"{self.output_dir}/predictive_ltv_results.csv"
        self.rfm.to_csv(output_path)
        print(f"Daten für das Dashboard gespeichert unter: {output_path}")

if __name__ == "__main__":
    modeler = PredictiveLTVModeler()
    modeler.load_and_prep_rfm_data()
    modeler.fit_models()
    modeler.finalize_predictions()