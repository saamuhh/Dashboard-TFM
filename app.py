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
    try:
        shap_muni = pd.read_csv('dashboard_shap_municipio.csv')
    except FileNotFoundError:
        shap_muni = None
    try:
        resumenes = pd.read_csv('resumenes_municipios.csv')
    except FileNotFoundError:
        resumenes = None
    try:
        enriquecido = pd.read_csv('datos_municipio_enriquecido.csv')
    except FileNotFoundError:
        enriquecido = None
    try:
        impacto_anom = pd.read_csv('dashboard_impacto_anomalias.csv')
    except FileNotFoundError:
        impacto_anom = None

    indicadores = ['TIT', 'TDT', 'IPH']
    escalador = StandardScaler().fit(historico[indicadores])
    pca = PCA(n_components=1).fit(escalador.transform(historico[indicadores]))
    pesos_pca = dict(zip(indicadores, pca.components_[0]))

    return historico, resumen, proyeccion, pesos_pca, shap_muni, resumenes, enriquecido, impacto_anom


historico, resumen, proyeccion, PESOS_PCA, shap_muni, resumenes, enriquecido, impacto_anom = cargar_datos()

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

tono_municipio = None
if enriquecido is not None and municipio in enriquecido['territorio'].values:
    fila_enr = enriquecido[enriquecido['territorio'] == municipio].iloc[0]
    tono_municipio = fila_enr.get('tono', None)

# ══════════════════════════════════════════════════════════════
# RESUMEN GENERADO POR IA (mT5 fine-tuneado)
# ══════════════════════════════════════════════════════════════

if resumenes is not None and municipio in resumenes['territorio'].values:
    import time
    texto_resumen = resumenes[resumenes['territorio'] == municipio]['resumen_llm'].iloc[0]

    with st.container(border=True):
        st.markdown("🧠 **Análisis del municipio** · generado por un modelo de lenguaje (mT5 fine-tuneado)")

        if st.session_state.get('ultimo_municipio') != municipio:
            st.session_state['ultimo_municipio'] = municipio

            def stream_texto():
                for palabra in texto_resumen.split(' '):
                    yield palabra + ' '
                    time.sleep(0.03)
            st.write_stream(stream_texto)
        else:
            st.write(texto_resumen)

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
    tono = tono_municipio if tono_municipio is not None else info.get('tono', None)
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

st.markdown("**Nivel de cada índice**")
g1, g2, g3 = st.columns(3)

GAUGE_CONFIG = {
    'TIT': ('Intensidad turística', g1),
    'TDT': ('Densidad turística', g2),
    'IPH': ('Presión humana', g3),
}

for ind, (nombre, col) in GAUGE_CONFIG.items():
    with col:
        q33 = historico[ind].quantile(0.33)
        q66 = historico[ind].quantile(0.66)
        vmax = historico[ind].quantile(0.98)
        valor = ultimo[ind]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=valor,
            title={'text': nombre, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, vmax]},
                'bar': {'color': '#333'},
                'steps': [
                    {'range': [0, q33], 'color': VERDE},
                    {'range': [q33, q66], 'color': '#EF9F27'},
                    {'range': [q66, vmax], 'color': CORAL},
                ],
            },
        ))
        fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

col_b, col_shap = st.columns(2)

with col_b:
    st.markdown("**Origen de las pernoctaciones**")
    hotel = serie['pernoctaciones'].sum()
    vv = serie['pernoc_vv'].sum()
    fig_orig = go.Figure(go.Bar(
        x=['Hotelero', 'Vivienda vacacional'], y=[hotel, vv],
        marker_color=[AZUL, CORAL],
    ))
    fig_orig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="Pernoctaciones acumuladas")
    st.plotly_chart(fig_orig, use_container_width=True)

with col_shap:
    st.markdown("**Factores que explican la presión (SHAP)**")
    if shap_muni is not None and municipio in shap_muni['territorio'].values:
        fila_shap = shap_muni[shap_muni['territorio'] == municipio].iloc[0]
        factores, valores = [], []
        for i in [1, 2, 3]:
            if pd.notna(fila_shap.get(f'factor_{i}_sube')):
                factores.append(fila_shap[f'factor_{i}_sube'])
                valores.append(fila_shap[f'factor_{i}_sube_valor'])
        for i in [1, 2, 3]:
            if pd.notna(fila_shap.get(f'factor_{i}_baja')):
                factores.append(fila_shap[f'factor_{i}_baja'])
                valores.append(fila_shap[f'factor_{i}_baja_valor'])

        orden = sorted(zip(factores, valores), key=lambda x: x[1])
        factores_o = [f for f, _ in orden]
        valores_o = [v for _, v in orden]

        fig_shap = go.Figure(go.Bar(
            x=valores_o, y=factores_o, orientation='h',
            marker_color=[CORAL if v > 0 else AZUL for v in valores_o],
        ))
        fig_shap.add_vline(x=0, line_color="#999")
        fig_shap.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Impacto en la predicción del TIT")
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Rojo: empuja la presión al alza · Azul: la modera")
    else:
        st.info("Sin datos SHAP disponibles para este municipio.")

col_radar, col_estac = st.columns(2)

with col_radar:
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
    fig_radar.update_layout(height=300, margin=dict(l=30, r=30, t=10, b=10),
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

with col_estac:
    st.markdown("**Estacionalidad de la presión**")
    estacional = (historico[historico['territorio'] == municipio]
                 .groupby(historico['fecha'].dt.month)['IPV'].mean())
    angulos = [m * 30 for m in estacional.index]
    fig_estac = go.Figure(go.Barpolar(
        r=estacional.values,
        theta=[MESES_CORTOS[m - 1] for m in estacional.index],
        marker_color=estacional.values,
        marker_colorscale=[[0, VERDE], [0.5, '#EF9F27'], [1, CORAL]],
    ))
    fig_estac.update_layout(height=300, margin=dict(l=30, r=30, t=10, b=10),
                            polar=dict(radialaxis=dict(showticklabels=False)))
    st.plotly_chart(fig_estac, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# DETECCIÓN DE ANOMALÍAS (AUTOENCODER LSTM)
# ══════════════════════════════════════════════════════════════

st.divider()
st.subheader("Detección de anomalías · Autoencoder LSTM")

if 'error' in historico.columns and historico['error'].notna().any():
    umbral = historico['error'].dropna().quantile(0.95)
    umbral_grave = umbral * 1.5

    anom_2025 = historico[historico['error'].notna()].copy()
    anom_2025['mes'] = anom_2025['fecha'].dt.month

    def gravedad(e):
        if e >= umbral_grave:
            return 2
        if e >= umbral:
            return 1
        return 0

    anom_2025['gravedad'] = anom_2025['error'].apply(gravedad)

    municipios_anomalos = (anom_2025[anom_2025['gravedad'] > 0]
                           .groupby('territorio')['gravedad'].sum()
                           .sort_values(ascending=False).head(10).index.tolist())

    if municipio not in municipios_anomalos and municipio in anom_2025['territorio'].values:
        if anom_2025[(anom_2025['territorio'] == municipio) & (anom_2025['gravedad'] > 0)].shape[0] > 0:
            municipios_anomalos = municipios_anomalos[:9] + [municipio]

    st.markdown("**Mapa de gravedad de anomalías en 2025** (meses × municipios más anómalos)")
    MESES_CORTOS_12 = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    matriz = []
    for terr in municipios_anomalos:
        fila_terr = []
        for m in range(1, 13):
            reg = anom_2025[(anom_2025['territorio'] == terr) & (anom_2025['mes'] == m)]
            fila_terr.append(reg['gravedad'].iloc[0] if len(reg) > 0 else 0)
        matriz.append(fila_terr)

    fig_heat = go.Figure(go.Heatmap(
        z=matriz, x=MESES_CORTOS_12, y=municipios_anomalos,
        colorscale=[[0, '#EAF3EA'], [0.5, '#EF9F27'], [1, CORAL]],
        showscale=True,
        colorbar=dict(title="Gravedad", tickvals=[0, 1, 2],
                      ticktext=['Normal', 'Leve', 'Grave']),
        zmin=0, zmax=2,
    ))
    fig_heat.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    fig_heat.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_heat, use_container_width=True)

    if impacto_anom is not None and municipio in impacto_anom['territorio'].values:
        imp = impacto_anom[impacto_anom['territorio'] == municipio].iloc[0]
        st.markdown(f"**Lectura de impacto · {municipio}**")

        n_anom = int(imp['n_meses_anomalos'])
        st.markdown(
            f"Este municipio presentó **{n_anom} meses anómalos** en 2025. "
            f"En esos meses, sus variables se situaron por encima de lo normal para el mes. "
            f"Si el patrón se mantuviera en 2026 (según la proyección del modelo), "
            f"cabría esperar los siguientes niveles:"
        )

        NOMBRES_VAR = {'pernoctaciones': 'Pernoctaciones',
                       'ingresos_totales': 'Ingresos',
                       'viajeros_alojados': 'Viajeros'}
        cols_imp = st.columns(3)
        for col_widget, (var, nombre) in zip(cols_imp, NOMBRES_VAR.items()):
            with col_widget:
                base = imp.get(f'{var}_2025', None)
                est = imp.get(f'{var}_2026_est', None)
                pct = imp.get(f'{var}_subida_pct', None)
                if pd.notna(est):
                    st.metric(nombre, f"{est:,.0f}", f"{pct:+.1f}% vs 2025")
        st.caption("Estimación condicional basada en el comportamiento observado en los meses "
                   "anómalos de 2025 y la proyección de presión para 2026. No implica causalidad.")
    else:
        if info.get('tiene_anomalia', False):
            st.info(f"{municipio} presenta anomalías en 2025, pero no hay estimación de impacto disponible.")
        else:
            st.success(f"{municipio} no presentó anomalías significativas en 2025.")
else:
    st.info("No hay datos de anomalías disponibles.")
