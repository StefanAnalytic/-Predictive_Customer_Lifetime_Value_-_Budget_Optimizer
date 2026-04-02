import subprocess
import sys
import time

def run_script(script_path, description):
    """Führt ein Python-Skript als Subprozess aus und fängt Fehler ab."""
    print(f"\n{'='*60}")
    print(f"🚀 Starte: {description}")
    print(f"📁 Datei: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Führt das Skript mit dem aktuellen Python-Interpreter aus
        subprocess.run([sys.executable, script_path], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Erfolgreich abgeschlossen in {elapsed_time:.2f} Sekunden.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei der Ausführung von {script_path}.")
        print("Bitte prüfe die obigen Fehlermeldungen.")
        sys.exit(1) # Bricht die Pipeline ab, wenn ein Schritt fehlschlägt

if __name__ == "__main__":
    print("🌟 Starte End-to-End LTV & Budget Optimization Pipeline 🌟\n")
    
    # Schritt 1: Daten generieren
    run_script("src/data_generator.py", "Phase 1: Synthetische Transaktionsdaten generieren (1.4M Rows)")
    
    # Schritt 2: Survival Analysis
    run_script("src/survival_analysis.py", "Phase 2 & 3: Survival Analysis & Churn-Wahrscheinlichkeiten")
    
    # Schritt 3: Predictive ML Models
    run_script("src/predictive_ltv.py", "Phase 4: BTYD Modellierung (BG/NBD & Gamma-Gamma)")
    
    print("🎉 PIPELINE ERFOLGREICH BEENDET! 🎉")
    print("Alle Modelle wurden trainiert und die Daten aufbereitet.")
    print("\n🌐 Starte interaktives C-Level Dashboard (Streamlit)...")
    print("Dein Browser sollte sich in wenigen Sekunden automatisch öffnen.\n")
    
    # Schritt 4: Dashboard automatisch öffnen
    try:
        # Wir rufen Streamlit direkt über das System auf
        subprocess.run(["streamlit", "run", "src/dashboard.py"])
    except KeyboardInterrupt:
        print("\nDashboard manuell beendet. Bis zum nächsten Mal!")
    except Exception as e:
        print(f"\n❌ Konnte Dashboard nicht automatisch starten. Fehler: {e}")
        print("Starte es manuell mit: streamlit run src/dashboard.py")