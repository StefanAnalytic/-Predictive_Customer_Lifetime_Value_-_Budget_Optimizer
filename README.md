# 🚀 Predictive Customer Lifetime Value & Budget Optimizer

Eine End-to-End Data Science Pipeline, die zukünftige Kundenwerte (pLTV) vorhersagt und eine nicht-lineare Budget-Optimierung für Marketing-Kanäle im E-Commerce durchführt.

## 🧠 Highlights & Architektur
- **Survival Analysis:** Kaplan-Meier-Schätzer zur Bestimmung der Kunden-"Lebenserwartung" und Identifizierung des Churn-Risikos.
- **Predictive ML (BTYD):** Einsatz von BG/NBD- und Gamma-Gamma-Modellen zur Vorhersage der latenten Kaufwahrscheinlichkeit ($P(Alive)$) und des zukünftigen LTVs im Non-Contractual Setting.
- **Prescriptive Analytics (Budget Optimizer):** Nicht-lineare Optimierung (`scipy.optimize`), um Marketing-Budgets unter Berücksichtigung von abnehmendem Grenznutzen (Diminishing Returns) so zu verteilen, dass die LTV:CAC-Ratio maximiert wird.
- **Robustes Engineering:** Vollständig kompatibel mit **Python 3.13** und **NumPy 2.0+** dank intelligenter Laufzeit-Patches (Monkey Patching) für ältere Machine-Learning-Bibliotheken.
- **C-Level Dashboard:** Eine interaktive `streamlit` Data App, die abstrakte Mathematik in strategische Business-Entscheidungen übersetzt.

## 🛠️ Tech Stack
Python 3.13 | `pandas` | `lifetimes` | `lifelines` | `scipy` | `streamlit` | `plotly`

---

## 🚀 Installation & One-Click Start

Das Projekt ist auf maximale Reproduzierbarkeit ausgelegt. Du musst keine einzelnen Skripte ausführen – ein einziger Befehl steuert die gesamte Datenverarbeitung.

cd ltv-portfolio-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Die Pipeline starten (One-Click): Führe nur diese eine Datei aus. Sie generiert automatisch 1,4 Millionen Transaktionsdaten, trainiert alle Modelle und öffnet am Ende das interaktive Streamlit-Dashboard in deinem Browser: python main.py