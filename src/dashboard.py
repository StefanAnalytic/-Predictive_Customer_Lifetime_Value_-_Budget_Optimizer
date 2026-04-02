import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Seite konfigurieren ---
st.set_page_config(page_title="LTV & Budget Optimizer", layout="wide")

st.title("🚀 Customer Lifetime Value & Budget Engine")
st.markdown("""
Dieses Dashboard kombiniert unsere **Predictive LTV Engine (BG/NBD)** mit Customer Acquisition Costs (CAC), 
um die **LTV:CAC Ratio** zu berechnen und optimale Budget-Umschichtungen zu empfehlen.
""")

# --- Daten laden ---
@st.cache_data
def load_predictive_data():
    try:
        df = pd.read_csv('data/processed/predictive_ltv_results.csv')
        return df
    except FileNotFoundError:
        st.error("Datei nicht gefunden. Bitte führe Phase 4 aus.")
        return None

df = load_predictive_data()

if df is not None:
    # 1. Aggregation des Vorhersage-Modells
    summary = df.groupby('acquisition_channel').agg(
        avg_expected_ltv=('predicted_6m_ltv', 'mean'),
        avg_prob_alive=('p_alive', 'mean'),
        customer_count=('customer_id', 'count')
    ).reset_index()

    # 2. BUSINESS LOGIK: Simulierte CAC (Customer Acquisition Cost) hinzufügen
    # In einem echten Projekt kämen diese Daten über eine API von Google Ads / Meta Ads.
    simulated_cac = {
        'Organic Search': 15.00,  # SEO ist "günstig" pro Kunde
        'Google Ads': 85.00,
        'Paid Social': 65.00,
        'Influencer': 120.00      # Influencer sind in der Akquise oft teuer
    }
    
    summary['cac'] = summary['acquisition_channel'].map(simulated_cac)
    
    # 3. DIE KÖNIGS-METRIKEN BERECHNEN
    summary['roi_percent'] = ((summary['avg_expected_ltv'] - summary['cac']) / summary['cac']) * 100
    summary['ltv_cac_ratio'] = summary['avg_expected_ltv'] / summary['cac']

    # --- KPI Sektion ---
    st.divider()
    st.subheader("🎯 Unit Economics (Prognose für die nächsten 6 Monate)")
    
    c1, c2, c3, c4 = st.columns(4)
    best_roi_channel = summary.loc[summary['roi_percent'].idxmax()]
    worst_roi_channel = summary.loc[summary['roi_percent'].idxmin()]
    
    c1.metric("Gesamtumsatz (Erwartet)", f"${df['predicted_6m_ltv'].sum():,.0f}")
    c2.metric("Höchster ROI Kanal", best_roi_channel['acquisition_channel'], f"{best_roi_channel['roi_percent']:.0f}%")
    c3.metric("Größter Kapitalvernichter", worst_roi_channel['acquisition_channel'], f"{worst_roi_channel['roi_percent']:.0f}%")
    c4.metric("Kunden analysiert", f"{len(df):,}")
    
    st.divider()

    # --- Visualisierungen ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### LTV vs. CAC (Die Profitabilitäts-Lücke)")
        fig_vs = go.Figure()
        fig_vs.add_trace(go.Bar(
            x=summary['acquisition_channel'],
            y=summary['avg_expected_ltv'],
            name='Erwarteter LTV ($)',
            marker_color='#1f77b4'
        ))
        fig_vs.add_trace(go.Bar(
            x=summary['acquisition_channel'],
            y=summary['cac'],
            name='Akquisitionskosten (CAC $)',
            marker_color='#ff7f0e'
        ))
        fig_vs.update_layout(barmode='group')
        st.plotly_chart(fig_vs, use_container_width=True)

    with col2:
        st.markdown("#### LTV:CAC Ratio (Gesundheit > 3.0)")
        fig_ratio = px.bar(
            summary.sort_values('ltv_cac_ratio', ascending=False),
            x='acquisition_channel',
            y='ltv_cac_ratio',
            color='ltv_cac_ratio',
            color_continuous_scale='RdYlGn', # Rot zu Grün Skala
            text_auto='.1f'
        )
        # Linie für die magische 3.0 Grenze
        fig_ratio.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Minimum Benchmark (3:1)")
        st.plotly_chart(fig_ratio, use_container_width=True)

    # --- Die Handlungsempfehlung für den CEO ---
    st.divider()
    st.markdown("### 💡 Strategische Handlungsempfehlung (Budget Shift)")
    
    st.info(f"""
    **Datenbasierte Empfehlung für das Q3/Q4 Marketing-Budget:**
    
    1. **Scale Up:** **{best_roi_channel['acquisition_channel']}** hat eine überragende LTV:CAC Ratio von **{best_roi_channel['ltv_cac_ratio']:.1f}**. Jeder investierte Dollar generiert langfristig einen enormen Profit. Das Budget hier sollte aggressiv erhöht werden, bis die CAC-Grenzkosten steigen.
    2. **Scale Down / Stop:** **{worst_roi_channel['acquisition_channel']}** verbrennt Kapital. Zwar liegt der initiale Bestellwert hoch, aber die Kunden wandern zu schnell ab (P(Alive) bei nur {(worst_roi_channel['avg_prob_alive']*100):.1f}%). Die Ratio von **{worst_roi_channel['ltv_cac_ratio']:.1f}** liegt massiv unter dem Industrie-Standard von 3.0. Budget hier einfrieren.
    """)

    st.subheader("Rohdaten: Unit Economics")
    st.dataframe(summary[['acquisition_channel', 'cac', 'avg_expected_ltv', 'ltv_cac_ratio', 'roi_percent']].style.format({
        'cac': '${:.2f}',
        'avg_expected_ltv': '${:.2f}',
        'ltv_cac_ratio': '{:.2f}x',
        'roi_percent': '{:.0f}%'
    }), use_container_width=True)
    
    # --- PHASE 8: PRESCRIPTIVE BUDGET OPTIMIZER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.header("🧮 Non-Linear Budget Optimizer (Prescriptive Analytics)")
    st.markdown("""
    Lineare Skalierung ist in der Realität eine Illusion. Jeder Kanal hat einen Sättigungspunkt (Diminishing Returns). 
    Hier nutzen wir nicht-lineare Optimierung (`scipy.optimize`), um das perfekte Budget-Portfolio zu berechnen. 
    Ziel: **Maximiere den Gesamt-LTV unter Einhaltung eines festen Budgets.**
    """)

    from scipy.optimize import minimize
    import numpy as np

    # Setup für den Optimizer
    total_budget = st.slider("💰 Setze das Quartals-Budget fest ($)", min_value=10000, max_value=500000, value=100000, step=10000)

    # Heuristik: Wie schnell sättigt ein Kanal? (Je kleiner der Alpha-Wert, desto schneller greifen Diminishing Returns)
    # Organic sättigt schnell (wenig Suchvolumen), Paid Social skaliert weit (viel Reichweite)
    channel_alphas = {
        'Organic Search': 0.6,
        'Google Ads': 0.8,
        'Paid Social': 0.85,
        'Influencer': 0.7
    }

    # Zielfunktion: Wir wollen den negativen LTV minimieren (was den echten LTV maximiert)
    # Formel für Diminishing Returns: Erwarteter Wert = LTV * (Spend^alpha) / Base_CAC
    def objective_function(spends):
        total_value = 0
        for i, channel in enumerate(summary['acquisition_channel']):
            ltv = summary.loc[i, 'avg_expected_ltv']
            base_cac = summary.loc[i, 'cac']
            alpha = channel_alphas[channel]
            
            # Neu gewonnene Kunden = (Spend^alpha) / (Base_CAC^alpha) -> Modelliert steigende CAC
            customers_acquired = (spends[i]**alpha) / (base_cac**alpha) if spends[i] > 0 else 0
            total_value += customers_acquired * ltv
            
        return -total_value # Negativ für Minimizer

    # Nebenbedingung: Summe der Ausgaben = Total Budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    
    # Grenzen: Kein negatives Budget (0 bis Total Budget pro Kanal)
    bounds = tuple((0, total_budget) for _ in range(len(summary)))
    
    # Startwerte: Gleichmäßige Verteilung
    initial_guess = [total_budget / len(summary)] * len(summary)

    # Optimierung starten
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Ergebnisse formatieren
    optimal_spends = result.x
    summary['optimal_budget'] = optimal_spends
    
    # KPI des Optimizers
    max_predicted_ltv = -result.fun
    budget_roi = ((max_predicted_ltv - total_budget) / total_budget) * 100

    c1, c2 = st.columns(2)
    c1.metric("Prognostizierter Return des Budgets", f"${max_predicted_ltv:,.0f}")
    c2.metric("Portfolio ROI (Nach Sättigungs-Effekten)", f"{budget_roi:,.0f}%")

    # Visualisierung der perfekten Aufteilung
    st.markdown("#### Die optimale Budget-Allokation")
    fig_opt = px.pie(
        summary, 
        values='optimal_budget', 
        names='acquisition_channel', 
        hole=0.4,
        title=f"Wie die {total_budget:,.0f} $ exakt verteilt werden sollten"
    )
    fig_opt.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_opt, use_container_width=True)

    # Der Action-Plan für den CEO
    st.info("📊 **Der CEO Action-Plan (inkl. Sättigungspolster):**")
    for i, row in summary.iterrows():
        spend = row['optimal_budget']
        if spend > 1000:
            st.write(f"- **{row['acquisition_channel']}:** Investiere exakt **${spend:,.0f}**. (Weiteres Budget würde hier die Grenzkosten über den profitablen LTV treiben).")
        else:
            st.write(f"- **{row['acquisition_channel']}:** Investiere **$0**. Dieser Kanal ist unter den aktuellen Bedingungen ein mathematischer Verlustbringer.")