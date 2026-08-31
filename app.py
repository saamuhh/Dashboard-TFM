import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Presión Turística Canarias", layout="wide")

COLOR_ALERTA = {'Alta': '#E24B4A', 'Media': '#EF9F27', 'Baja': '#639922'}
NOMBRE_CLUSTER = {0: 'Destino turístico', 1: 'Capital urbana', 2: 'Baja presión'}
AZUL, CORAL, VERDE = '#378ADD', '#E24B4A', '#639922'
MESES_NOMBRE = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
MESES_CORTOS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']


@st.cache_data
def cargar_datos():
    historico = pd.read_csv('dashboard_historico.csv', parse_dates=['fecha'])
    resumen = pd.read_csv('dashboard_resumen.csv')
    try:
        proyeccion = pd.read_csv('dashboard_proyeccion_2026.csv', parse_dates=['fecha'])
    except FileNotFoundError:
        proyeccion = None

    indicadores = ['TIT', 'TDT', 'IPH']
    escalador = StandardScaler().fit(historico[indicadores])
    pca = PCA(n_components=1).fit(escalador.transform(historico[indicadores]))
    pesos_pca = dict(zip(indicadores, pca.components_[0]))

    return historico, resumen, proyeccion, pesos_pca


historico, resumen, proyeccion, PESOS_PCA = cargar_datos()

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

# ══════════════════════════════════════════════════════════════
# TARJETAS DE ESTADO
# ══════════════════════════════════════════════════════════════

ranking = resumen['IPV_actual'].rank(ascending=False)
posicion = int(ranking[resumen['territorio'] == municipio].iloc[0])

proy_muni = None
if proyeccion is not None:
    proy_muni = proyeccion[proyeccion['territorio'] == municipio].sort_values('fecha')

st.markdown("**Estado actual**")
f1c1, f1c2, f1c3, f1c4 = st.columns(4)
with f1c1:
    color = COLOR_ALERTA[info['alerta']]
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Nivel de alerta</p>"
                f"<p style='color:{color};font-size:26px;font-weight:600;margin:0'>{info['alerta']}</p>",
                unsafe_allow_html=True)
with f1c2:
    st.metric("IPV actual", f"{info['IPV_actual']:.2f}")
with f1c3:
    st.metric("Posición", f"{posicion}º de 88")
with f1c4:
    st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Cluster</p>"
                f"<p style='font-size:18px;font-weight:600;margin:0'>"
                f"{NOMBRE_CLUSTER.get(int(info['cluster']), 'N/D')}</p>",
                unsafe_allow_html=True)

st.markdown("**Predicción 2026**")
f2c1, f2c2, f2c3, f2c4, f2c5 = st.columns(5)

if proy_muni is not None and len(proy_muni) > 0:
    ipv_2026_medio = proy_muni['IPV_central'].mean()
    variacion = (ipv_2026_medio - info['IPV_actual']) / (abs(info['IPV_actual']) + 1e-9) * 100
    mes_pico_idx = int(proy_muni.reset_index()['IPV_central'].idxmax())
    mes_pico = MESES_NOMBRE[proy_muni.iloc[mes_pico_idx]['fecha'].month - 1]
    amplitud = (proy_muni['IPV_sup'] - proy_muni['IPV_inf']).mean()
    incert = 'Alta' if amplitud > 2 else ('Media' if amplitud > 1 else 'Baja')

    with f2c1:
        st.metric("Proyección 2026", f"{ipv_2026_medio:.2f}")
    with f2c2:
        st.metric("Variación 25→26", f"{variacion:+.0f}%")
    with f2c3:
        st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Mes de mayor presión</p>"
                    f"<p style='font-size:18px;font-weight:600;margin:0;text-transform:capitalize'>{mes_pico}</p>",
                    unsafe_allow_html=True)
    with f2c4:
        color_i = {'Alta': '#E24B4A', 'Media': '#EF9F27', 'Baja': '#639922'}[incert]
        st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Incertidumbre</p>"
                    f"<p style='color:{color_i};font-size:18px;font-weight:600;margin:0'>{incert}</p>",
                    unsafe_allow_html=True)
else:
    for c in [f2c1, f2c2, f2c3, f2c4]:
        c.caption("Sin proyección")

with f2c5:
    tono = info.get('tono', None)
    if pd.notna(tono):
        color_t = {'crítico': '#E24B4A', 'neutro': '#888', 'positivo': '#639922'}.get(tono, '#888')
        st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Tono mediático</p>"
                    f"<p style='color:{color_t};font-size:18px;font-weight:600;margin:0;text-transform:capitalize'>{tono}</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:#888;font-size:13px;margin:0'>Tono mediático</p>"
                    f"<p style='color:#bbb;font-size:14px;margin:0'>N/D</p>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# EVOLUCIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════
# DESCOMPOSICIÓN DE LA PRESIÓN
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Descomposición de la presión")

medias_ind = {ind: historico[ind].mean() for ind in PESOS_PCA}
stds_ind = {ind: historico[ind].std() for ind in PESOS_PCA}

ultimo = serie.iloc[-1] if len(serie) > 0 else historico[historico['territorio'] == municipio].iloc[-1]

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Aportación de cada indicador al IPV**")
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
    fig_desc.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Veces la media de Canarias")
    st.plotly_chart(fig_desc, use_container_width=True)

with col_b:
    st.markdown("**Origen de las pernoctaciones**")
    hotel = serie['pernoctaciones'].sum()
    vv = serie['pernoc_vv'].sum()
    fig_orig = go.Figure(go.Bar(
        x=['Hotelero', 'Vivienda vacacional'], y=[hotel, vv],
        marker_color=[AZUL, CORAL],
    ))
    fig_orig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Pernoctaciones acumuladas")
    st.plotly_chart(fig_orig, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Composición real del IPV (pesos PCA)**")
    contrib_pca = {}
    for ind in PESOS_PCA:
        z = (ultimo[ind] - medias_ind[ind]) / stds_ind[ind]
        contrib_pca[ind] = z * PESOS_PCA[ind]
    total_abs = sum(abs(v) for v in contrib_pca.values()) + 1e-9
    nombres_ind = {'TIT': 'Intensidad', 'TDT': 'Densidad', 'IPH': 'Presión humana'}
    fig_pca = go.Figure(go.Bar(
        x=[contrib_pca[i] for i in PESOS_PCA], y=[nombres_ind[i] for i in PESOS_PCA],
        orientation='h',
        marker_color=[CORAL if v > 0 else AZUL for v in contrib_pca.values()],
        text=[f"{abs(v)/total_abs:.0%}" for v in contrib_pca.values()],
        textposition='outside',
    ))
    fig_pca.add_vline(x=0, line_color="#999")
    fig_pca.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Contribución al IPV")
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(f"IPV = {ultimo['IPV']:.2f} · suma de las tres contribuciones ponderadas por PCA")

with col_d:
    st.markdown("**Perfil vs media del cluster**")
    muni_cluster_ids = resumen[resumen['cluster'] == info['cluster']]['territorio']
    media_cluster_ind = historico[historico['territorio'].isin(muni_cluster_ids)][list(PESOS_PCA)].mean()

    categorias = ['Intensidad (TIT)', 'Densidad (TDT)', 'Presión humana (IPH)']
    valores_muni = [(ultimo[i] - historico[i].min()) / (historico[i].max() - historico[i].min() + 1e-9)
                    for i in PESOS_PCA]
    valores_cluster = [(media_cluster_ind[i] - historico[i].min()) / (historico[i].max() - historico[i].min() + 1e-9)
                       for i in PESOS_PCA]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=valores_muni + [valores_muni[0]],
                                        theta=categorias + [categorias[0]],
                                        fill='toself', name=municipio,
                                        line=dict(color=CORAL)))
    fig_radar.add_trace(go.Scatterpolar(r=valores_cluster + [valores_cluster[0]],
                                        theta=categorias + [categorias[0]],
                                        fill='toself', name='Media cluster',
                                        line=dict(color='#999', dash='dot')))
    fig_radar.update_layout(height=280, margin=dict(l=30, r=30, t=10, b=10),
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("**Estacionalidad de la presión**")
estacional = (historico[historico['territorio'] == municipio]
             .groupby(historico['fecha'].dt.month)['IPV'].mean())
fig_estac = go.Figure(go.Bar(
    x=[MESES_CORTOS[m - 1] for m in estacional.index], y=estacional.values,
    marker_color=AZUL,
))
fig_estac.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="IPV medio")
st.plotly_chart(fig_estac, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# DETECCIÓN DE ANOMALÍAS (AUTOENCODER LSTM)
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Detección de anomalías · Autoencoder LSTM")

col_err, col_rank = st.columns([3, 2])

with col_err:
    st.markdown("**Error de reconstrucción mensual**")
    serie_anom = serie[serie['error'].notna()].sort_values('fecha') if 'error' in serie.columns else pd.DataFrame()
    if len(serie_anom) > 0:
        umbral_muni = historico['error'].dropna().quantile(0.95)
        colores_barras = [CORAL if e > umbral_muni else AZUL for e in serie_anom['error']]
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(x=serie_anom['fecha'], y=serie_anom['error'],
                                 marker_color=colores_barras, name='Error'))
        fig_err.add_hline(y=umbral_muni, line_dash="dash", line_color="#666",
                          annotation_text="Umbral de anomalía")
        fig_err.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="Error de reconstrucción")
        st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("Este municipio no tiene datos de anomalías (solo se calculan sobre 2025).")

with col_rank:
    st.markdown("**Municipios más anómalos en 2025**")
    if 'anomalia' in historico.columns:
        ranking_anom = (historico[historico['anomalia'] == True]
                        .groupby('territorio').size()
                        .sort_values(ascending=False).head(10))
        if len(ranking_anom) > 0:
            colores_rank = [CORAL if t == municipio else '#B0B0B0' for t in ranking_anom.index]
            fig_ra = go.Figure(go.Bar(
                x=ranking_anom.values, y=ranking_anom.index, orientation='h',
                marker_color=colores_rank,
            ))
            fig_ra.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                 xaxis_title="Meses anómalos")
            fig_ra.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_ra, use_container_width=True)

if info.get('tiene_anomalia', False):
    st.warning(f"Anomalía detectada por el autoencoder: la presión de {municipio} "
               f"se desvía de su patrón histórico en 2025.", icon="⚠️")
