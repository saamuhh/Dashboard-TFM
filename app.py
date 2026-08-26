import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Presión Turística Canarias", layout="wide")

COLOR_ALERTA = {'Alta': '#E24B4A', 'Media': '#EF9F27', 'Baja': '#639922'}
NOMBRE_CLUSTER = {0: 'Destino turístico', 1: 'Capital urbana', 2: 'Baja presión'}


@st.cache_data
def cargar_datos():
    historico = pd.read_csv('dashboard_historico.csv', parse_dates=['fecha'])
    resumen = pd.read_csv('dashboard_resumen.csv')
    return historico, resumen


historico, resumen = cargar_datos()

st.title("Sistema de alerta de presión turística")
st.caption("Canarias · nivel municipal · datos ISTAC 2021–2025")

municipio = st.selectbox("Municipio", sorted(resumen['territorio'].unique()))
info = resumen[resumen['territorio'] == municipio].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    color = COLOR_ALERTA[info['alerta']]
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Nivel de alerta</p>"
                f"<p style='color:{color};font-size:26px;font-weight:600;margin:0'>{info['alerta']}</p>",
                unsafe_allow_html=True)

with col2:
    st.metric("IPV actual", f"{info['IPV_actual']:.2f}")

with col3:
    delta = info['IPV_pred'] - info['IPV_actual']
    st.metric("Predicción próx. mes", f"{info['IPV_pred']:.2f}", f"{delta:+.2f}")

with col4:
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Cluster</p>"
                f"<p style='font-size:18px;font-weight:600;margin:0'>"
                f"{NOMBRE_CLUSTER.get(int(info['cluster']), 'N/D')}</p>",
                unsafe_allow_html=True)

st.divider()
col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("Ranking de presión")
    top = resumen.nlargest(15, 'IPV_actual').sort_values('IPV_actual')
    fig_rank = go.Figure(go.Bar(
        x=top['IPV_actual'], y=top['territorio'], orientation='h',
        marker_color=[COLOR_ALERTA[a] for a in top['alerta']],
    ))
    fig_rank.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="IPV actual")
    st.plotly_chart(fig_rank, use_container_width=True)

with col_der:
    st.subheader("Evolución del IPV")
    serie = historico[historico['territorio'] == municipio].sort_values('fecha')
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(x=serie['fecha'], y=serie['IPV'],
                                mode='lines', name='Histórico',
                                line=dict(color='#378ADD', width=2)))
    fig_ev.add_trace(go.Scatter(x=[serie['fecha'].iloc[-1]], y=[info['IPV_pred']],
                                mode='markers', name='Predicción',
                                marker=dict(color='#E24B4A', size=10)))
    fig_ev.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                         yaxis_title="IPV")
    st.plotly_chart(fig_ev, use_container_width=True)

if info['tiene_anomalia']:
    st.warning(f"Anomalía detectada por el autoencoder: la presión de {municipio} "
               f"se desvía de su patrón histórico en 2025.", icon="⚠️")
