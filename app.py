import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Presión Turística Canarias", layout="wide")

COLOR_ALERTA = {'Alta': '#E24B4A', 'Media': '#EF9F27', 'Baja': '#639922'}
NOMBRE_CLUSTER = {0: 'Destino turístico', 1: 'Capital urbana', 2: 'Baja presión'}
AZUL, CORAL, VERDE = '#378ADD', '#E24B4A', '#639922'


@st.cache_data
def cargar_datos():
    historico = pd.read_csv('dashboard_historico.csv', parse_dates=['fecha'])
    resumen = pd.read_csv('dashboard_resumen.csv')
    try:
        proyeccion = pd.read_csv('dashboard_proyeccion_2026.csv', parse_dates=['fecha'])
    except FileNotFoundError:
        proyeccion = None
    return historico, resumen, proyeccion


historico, resumen, proyeccion = cargar_datos()

st.title("Cuadro de mando · Presión turística en Canarias")
st.caption("Monitorización municipal · datos ISTAC 2021–2025 · proyección 2026")

col_muni, col_rango = st.columns([1, 2])
with col_muni:
    municipio = st.selectbox("Municipio", sorted(resumen['territorio'].unique()))
with col_rango:
    años = sorted(historico['fecha'].dt.year.unique())
    rango = st.select_slider("Rango temporal", options=años, value=(años[0], años[-1]))

info = resumen[resumen['territorio'] == municipio].iloc[0]
serie = historico[(historico['territorio'] == municipio) &
                  (historico['fecha'].dt.year >= rango[0]) &
                  (historico['fecha'].dt.year <= rango[1])].sort_values('fecha')

ranking = resumen['IPV_actual'].rank(ascending=False)
posicion = int(ranking[resumen['territorio'] == municipio].iloc[0])

c1, c2, c3, c4 = st.columns(4)
with c1:
    color = COLOR_ALERTA[info['alerta']]
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Nivel de alerta</p>"
                f"<p style='color:{color};font-size:26px;font-weight:600;margin:0'>{info['alerta']}</p>",
                unsafe_allow_html=True)
with c2:
    st.metric("IPV actual", f"{info['IPV_actual']:.2f}")
with c3:
    st.metric("Posición", f"{posicion}º de 88")
with c4:
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Cluster</p>"
                f"<p style='font-size:18px;font-weight:600;margin:0'>"
                f"{NOMBRE_CLUSTER.get(int(info['cluster']), 'N/D')}</p>",
                unsafe_allow_html=True)

st.divider()
st.subheader("Evolución temporal")

fig_ipv = go.Figure()
fig_ipv.add_trace(go.Scatter(x=serie['fecha'], y=serie['IPV'],
                             mode='lines', name='IPV histórico',
                             line=dict(color=AZUL, width=2)))

if proyeccion is not None:
    proy = proyeccion[proyeccion['territorio'] == municipio].sort_values('fecha')
    fig_ipv.add_trace(go.Scatter(x=proy['fecha'], y=proy['IPV_sup'],
                                 mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'))
    fig_ipv.add_trace(go.Scatter(x=proy['fecha'], y=proy['IPV_inf'],
                                 mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(226,75,74,0.15)',
                                 name='Banda 80%', hoverinfo='skip'))
    fig_ipv.add_trace(go.Scatter(x=proy['fecha'], y=proy['IPV_central'],
                                 mode='lines', name='Proyección 2026',
                                 line=dict(color=CORAL, width=2, dash='dash')))

fig_ipv.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="IPV")
st.plotly_chart(fig_ipv, use_container_width=True)

col_ind, col_cluster = st.columns(2)

with col_ind:
    st.markdown("**Indicadores normalizados**")
    fig_ind = go.Figure()
    for col, nombre, color in [('TIT', 'Intensidad', CORAL),
                               ('TDT', 'Densidad', AZUL),
                               ('IPH', 'Presión humana', VERDE)]:
        norm = (serie[col] - serie[col].min()) / (serie[col].max() - serie[col].min() + 1e-9)
        fig_ind.add_trace(go.Scatter(x=serie['fecha'], y=norm, mode='lines',
                                     name=nombre, line=dict(color=color, width=2)))
    fig_ind.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Normalizado 0–1")
    st.plotly_chart(fig_ind, use_container_width=True)

with col_cluster:
    st.markdown("**Municipio vs media de su cluster**")
    cluster_muni = info['cluster']
    muni_cluster = resumen[resumen['cluster'] == cluster_muni]['territorio']
    serie_cluster = (historico[historico['territorio'].isin(muni_cluster)]
                     .groupby('fecha')['IPV'].mean().reset_index())
    serie_cluster = serie_cluster[(serie_cluster['fecha'].dt.year >= rango[0]) &
                                  (serie_cluster['fecha'].dt.year <= rango[1])]
    fig_cl = go.Figure()
    fig_cl.add_trace(go.Scatter(x=serie['fecha'], y=serie['IPV'], mode='lines',
                                name=municipio, line=dict(color=CORAL, width=2)))
    fig_cl.add_trace(go.Scatter(x=serie_cluster['fecha'], y=serie_cluster['IPV'],
                                mode='lines', name='Media cluster',
                                line=dict(color='#999', width=2, dash='dot')))
    fig_cl.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="IPV")
    st.plotly_chart(fig_cl, use_container_width=True)

st.divider()
st.subheader("Descomposición de la presión")

col_desc1, col_desc2 = st.columns(2)

with col_desc1:
    st.markdown("**Aportación de cada indicador al IPV**")
    ultimo = serie.iloc[-1]
    contribuciones = {'Intensidad (TIT)': ultimo['TIT'],
                      'Densidad (TDT)': ultimo['TDT'],
                      'Presión humana (IPH)': ultimo['IPH']}
    medias_canarias = {'Intensidad (TIT)': historico['TIT'].mean(),
                       'Densidad (TDT)': historico['TDT'].mean(),
                       'Presión humana (IPH)': historico['IPH'].mean()}
    ratios = {k: contribuciones[k] / medias_canarias[k] for k in contribuciones}
    fig_desc = go.Figure(go.Bar(
        x=list(ratios.values()), y=list(ratios.keys()), orientation='h',
        marker_color=[CORAL if v > 1 else AZUL for v in ratios.values()],
    ))
    fig_desc.add_vline(x=1, line_dash="dash", line_color="#666",
                       annotation_text="Media Canarias")
    fig_desc.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Veces la media de Canarias")
    st.plotly_chart(fig_desc, use_container_width=True)

with col_desc2:
    st.markdown("**Origen de las pernoctaciones**")
    serie_o = serie.copy()
    hotel = serie_o['pernoctaciones'].sum()
    vv = serie_o['pernoc_vv'].sum()
    fig_orig = go.Figure(go.Bar(
        x=['Hotelero', 'Vivienda vacacional'], y=[hotel, vv],
        marker_color=[AZUL, CORAL],
    ))
    fig_orig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Pernoctaciones acumuladas")
    st.plotly_chart(fig_orig, use_container_width=True)

if info['tiene_anomalia']:
    st.warning(f"Anomalía detectada por el autoencoder: la presión de {municipio} "
               f"se desvía de su patrón histórico en 2025.", icon="⚠️")
