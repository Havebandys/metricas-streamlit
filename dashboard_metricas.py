# dashboard_metricas.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Dashboard de Métricas de Trading", layout="wide")

# Ajuste de estilo global para gráficos
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.facecolor': 'white',
    'axes.grid': False,
})

# Paleta pastel con sensación ejecutiva
PALETTE = [
    '#AEC6CF',  # Azul pastel
    '#FFB347',  # Naranja pastel
    '#779ECB',  # Azul cielo claro
    '#F49AC2',  # Rosa pastel
    '#B19CD9'   # Lila pastel
]

# 1. Carga de datos (solo Excel .xlsx)
st.sidebar.header("1. Carga de datos")
archivo = st.sidebar.file_uploader("Sube tu archivo de operaciones (.xlsx)", type=["xlsx"])
if not archivo:
    st.warning("Por favor, sube un archivo .xlsx para continuar.")
    st.stop()

# Leer datos
df = pd.read_excel(archivo, parse_dates=["fecha_entrada", "fecha_salida"])

# Verificar columnas obligatorias
columnas_obligatorias = [
    "id_operacion", "fecha_entrada", "fecha_salida",
    "activo", "lado", "precio_entrada", "precio_salida",
    "probabilidad_asignada"
]
faltantes = [c for c in columnas_obligatorias if c not in df.columns]
if faltantes:
    st.error(f"Faltan columnas obligatorias: {faltantes}")
    st.stop()

# 2. Cálculo de retorno y resultado
def calcular_retorno(fila):
    if str(fila["lado"]).lower() == "largo":
        return (fila["precio_salida"] - fila["precio_entrada"]) / fila["precio_entrada"]
    return (fila["precio_entrada"] - fila["precio_salida"]) / fila["precio_entrada"]

df["retorno_realizado"] = df.apply(calcular_retorno, axis=1)
df["resultado"] = (df["retorno_realizado"] > 0).astype(int)

# Pre-cálculos para gráficos
datos_probs = df["probabilidad_asignada"].clip(0,1)
bins = np.linspace(0,1,6)
centros = (bins[:-1] + bins[1:]) / 2
frecuencia_real, _, _ = binned_statistic(datos_probs, df["resultado"], "mean", bins=bins)
retornos = df["retorno_realizado"]
tamanios = pd.to_numeric(df.get("tamanio", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
serie_equity = pd.Series((retornos * tamanios).cumsum().values, index=df["fecha_salida"].sort_values())

# 3. Métricas principales
st.title("📊 Dashboard de Métricas de Trading")
col1, col2, col3 = st.columns(3)

# Resumen de desempeño
with col1:
    st.subheader("Resumen de desempeño")
    total_ops = len(df)
    tasa_exito = df["resultado"].mean()
    gan_media = retornos[retornos > 0].mean()
    per_media = retornos[retornos <= 0].mean()
    expectativa = tasa_exito * gan_media + (1 - tasa_exito) * per_media
    st.metric("Operaciones totales", total_ops)
    st.metric("Tasa de éxito", f"{tasa_exito:.1%}")
    st.metric("Expectativa/op.", f"{expectativa:.2%}")
    st.metric("Gan. media (+)", f"{gan_media:.2%}")
    st.metric("Pérd. media (–)", f"{per_media:.2%}")

# Curva de calibración
with col2:
    st.subheader("Curva de calibración")
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(centros, frecuencia_real, "o-", color=PALETTE[0], linewidth=1)
    ax1.plot([0,1],[0,1], "--", color=PALETTE[1], linewidth=1)
    ax1.set_title("Curva de calibración", fontsize=8)
    ax1.set_xlabel("Prob. asignada", fontsize=7)
    ax1.set_ylabel("Frecuencia real", fontsize=7)
    ax1.legend(frameon=False)
    ax1.tick_params(labelsize=6)
    st.pyplot(fig1)
    brier = np.mean((df["probabilidad_asignada"] - df["resultado"]) ** 2)
    st.write(f"**Brier Score:** {brier:.4f}")

# Adherencia al proceso
with col3:
    st.subheader("Adherencia al proceso")
    cols_bool = [c for c in ["regla_riesgo_ok","regla_salida_ok","tesis_documentada","proceso_seguido"] if c in df.columns]
    if cols_bool:
        adher = {c: df[c].mean() for c in cols_bool}
        df_ad = pd.DataFrame.from_dict(adher, orient="index", columns=["Cumplimiento"]).round(3)
        st.table(df_ad.style.format("{:.1%}"))
    else:
        st.info("No se detectaron columnas de control de proceso.")

# 4. Gráficos media pantalla
st.markdown("---")
row1, row2 = st.columns(2)

# Histograma de retornos
with row1:
    st.subheader("Histograma de retornos")
    fig2, ax2 = plt.subplots(figsize=(5,3))
    n, bins2, patches = ax2.hist(retornos, bins=15, color=PALETTE[2], edgecolor='white', linewidth=0.7)
    media, desv = retornos.mean(), retornos.std()
    x = np.linspace(retornos.min(), retornos.max(), 100)
    gauss = (1/(desv*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-media)/desv)**2)
    gauss = gauss * max(n)/max(gauss)
    ax2.plot(x, gauss, '--', color=PALETTE[3], linewidth=1)
    ax2.set_title("Retornos vs Gauss", fontsize=8)
    ax2.set_xlabel("Retorno", fontsize=7)
    ax2.tick_params(labelsize=6)
    st.pyplot(fig2)

# Sobre-slippage
with row2:
    st.subheader("Sobre-slippage")
    slip_df = df.dropna(subset=["slippage_estimado_pct","slippage_real_pct"])
    slip_vals = slip_df['slippage_real_pct'] - slip_df['slippage_estimado_pct']
    fig3, ax3 = plt.subplots(figsize=(5,3))
    ax3.bar(range(len(slip_vals)), slip_vals, color=PALETTE[4], edgecolor='white', linewidth=0.7)
    ax3.set_title("Sobre-slippage", fontsize=8)
    ax3.set_xlabel("Operaciones", fontsize=7)
    ax3.tick_params(labelsize=6)
    st.pyplot(fig3)

# 5. Equity y drawdown
st.markdown("---")
eq_col, dd_col = st.columns(2)
with eq_col:
    st.subheader("Equity acumulado")
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(serie_equity.index, serie_equity.values, color=PALETTE[0], linewidth=1)
    ax4.set_title("Equity acumulado", fontsize=8)
    ax4.set_xlabel("Fecha", fontsize=7)
    ax4.tick_params(labelsize=6)
    st.pyplot(fig4)
with dd_col:
    st.subheader("Drawdown")
    fig5, ax5 = plt.subplots(figsize=(5,3))
    peak = serie_equity.cummax()
    dd = (serie_equity - peak)/peak
    ax5.fill_between(dd.index, dd.values, color=PALETTE[1], alpha=0.3)
    ax5.set_title("Drawdown", fontsize=8)
    ax5.set_xlabel("Fecha", fontsize=7)
    ax5.tick_params(labelsize=6)
    st.pyplot(fig5)

# 6. Detalle de operaciones
st.markdown("---")
st.subheader("Detalle de operaciones")
cols_det = [
    "id_operacion","fecha_entrada","fecha_salida",
    "activo","lado","precio_entrada","precio_salida",
    "retorno_realizado","probabilidad_asignada"
] + cols_bool
df_det = df[cols_det].copy()
# Formatear fechas en dd/mm/YYYY para pantalla
for col in ['fecha_entrada', 'fecha_salida']:
    df_det[col] = pd.to_datetime(df_det[col]).dt.strftime('%d/%m/%Y')
df_det['retorno_realizado'] = (df_det['retorno_realizado']*100).round(1).astype(str)+'%'
# Aplicar estilo: centrar celdas y achicar fuente
df_estilado = df_det.sort_values('fecha_salida', ascending=False)\
    .style.set_properties(**{
        'text-align': 'center',
        'font-size': '12px',
        'padding': '4px'
    })
# Mostrar tabla estilizada
st.write(df_estilado, unsafe_allow_html=True)

# 7. Alertas automáticas
st.markdown("---")
st.subheader("⚠️ Alertas")
alerts = []
if len(centros) >= 3 and abs(frecuencia_real[2] - centros[2]) > 0.15:
    alerts.append("Desviación >15% en calibración (prob ~50%)")
for regla, cumplimiento in {c: df[c].mean() for c in cols_bool}.items():
    if cumplimiento < 0.7:
        alerts.append(f"Adherencia baja en '{regla}': {cumplimiento:.0%}")
if expectativa < 0:
    alerts.append("Expectativa por operación negativa")
if alerts:
    for msg in alerts:
        st.warning(msg)
else:
    st.success("Sin alertas críticas")

# 8. Glosario de Alertas
st.markdown("---")
st.subheader("📖 Glosario de Alertas")
# Texto multilínea para evitar problemas de comillas
glosario = """
1. **D⚠️ Desviación en calibración (>15 % en prob ~50 %)

Compara tu tasa real de éxitos para operaciones a las que les asignaste cerca del 50 % de probabilidad (el “bin medio”) con la probabilidad ideal (50 %).

Si la frecuencia real difiere más de 15 puntos porcentuales de ese 50 %, indica que tu modelo de probabilidades no está bien calibrado en torno al nivel medio.

Interpretación: revisá tu sistema de estimación (por ejemplo, ajustes de riesgo, inputs o sesgos) para asegurarte de que cuando dices “50 %” tus trades ganan efectivamente cerca de la mitad de las veces.

2. **⚠️ Adherencia baja en ‘regla_…’ (<70 %)

Por cada control de proceso que tengas (regla_riesgo_ok, regla_salida_ok, tesis_documentada, proceso_seguido), medimos el % de operaciones donde cumpliste esa regla.

Si alguna queda por debajo del 70 %, significa que en al menos 3 de cada 10 trades no seguiste esa guía: por ejemplo, no documentaste la tesis o no respetaste el stop-loss.

Interpretación: una baja adherencia suele traducirse en mayor variabilidad y riesgo inesperado. Conviene reforzar disciplina o ajustar tu checklist.

3. **⚠️ Expectativa por operación negativa

La expectativa es el valor esperado de retorno (win_rate * gan_media + loss_rate * per_media).

Si es negativa, indica que tu sistema (con tu tasa de éxito y tamaño de ganancia/pérdida) no debería, en promedio, ser rentable.

Interpretación: revisá el ratio ganancia/pérdida o tu win rate; quizá necesitas redefinir tamaños de posición, mejorar selección de setups o incorporar filtros para elevar tu expectativa.

4. **✅ Sin alertas críticas

Ninguna de las condiciones anteriores se cumplió: tu calibración, adherencia y expectativa están dentro de los umbrales deseados.

Interpretación: sigue monitoreando, pero tu sistema funciona de acuerdo con las reglas y proyecciones establecidas.

5. **⚠️ Sobre-slippage

Qué muestra:
La diferencia entre el slippage real (precio de ejecución) y el slippage estimado en cada operación.

Cómo leerlo:

Barras verticales: cada barra es una operación; la altura es el “exceso” de slippage en porcentaje.

Qué indica un buen desempeño de ejecución:

Valores cercanos a cero indican que tu estimación de slippage fue precisa.

Señales de alerta:

Barras consistentemente por encima de cero (slippage real > estimado) indican costos imprevistos de ejecución.

6. **📈💹 Equity acumulado

Equity Acumulado
Qué muestra:
La evolución acumulada de P&L (ganancias o pérdidas) a lo largo del tiempo, ordenada por fecha de cierre.

Cómo leerlo:

Una curva ascendente constante implica crecimiento de capital.

Tramos planos indican periodos sin ganancias netas; descensos, pérdidas.

Qué indica un buen desempeño:

Una pendiente positiva sostenida.

Pocas caídas abruptas.

7. **📉🔻 Drawdown máximo

Qué muestra:
El retroceso porcentual desde el máximo histórico de la curva de equity hasta cada punto en el tiempo.

Cómo leerlo:

Cada punto es la caída actual respecto al pico previo.

Valores de –5 % a –10 % son caídas menores; –20 %, caídas significativas.

Qué indica un buen perfil de riesgo:

Drawdowns moderados y de corta duración.

Rápida recuperación hacia nuevos picos.

Señales de alerta:

Drawdowns profundos (> 15 %) o prolongados indican que tu estrategia puede exponer mucho capital al riesgo

8. **📊 Ratio de Sharpe

Qué muestra:
El ratio de Sharpe mide el rendimiento ajustado al riesgo de tu estrategia, comparando la rentabilidad excesiva (sobre la tasa libre de riesgo) con la volatilidad de los retornos.

Cómo leerlo:
Un ratio de Sharpe más alto indica un mejor rendimiento ajustado al riesgo.

Qué indica un buen desempeño:
Un ratio de Sharpe superior a 1 es generalmente considerado bueno; superior a 2, excelente.

9. **🎯🔄 Curva de calibración

Qué muestra:
Compara la probabilidad que asignaste a cada trade (por ejemplo, 10 %, 20 %, …, 90 %) con la tasa real de éxito en ese rango.

Cómo leerla:

La línea punteada (ideal) es la bisectriz: si dices “20 %” de probabilidad, deberías ganar el 20 % de esas operaciones.

Si el punto real (círculo) está sobre la bisectriz, ganas más de lo que esperabas; si está por debajo, ganas menos.

Qué indica una buena calibración:

Los puntos alineados cerca de la línea ideal en todo el rango de probabilidades.

Desviaciones pequeñas (< ±5 %) indican que tu modelo de probabilidad es fiable.

Señales de alerta:

Un punto desviado > 15 % en el bin medio (prob ~ 50 %) sugiere que tu probabilidad “50 %” no refleja bien la mitad de éxitos.

10. **📊🔢 Histograma de Retornos vs. Campana Gaussiana
Qué muestra:

Barras: distribución empírica de retornos (ganancias/pérdidas por trade).

Línea punteada: ajuste teórico de una distribución normal (campana de Gauss) con la misma media y desviación estándar.

Cómo leerlo:

Compara la forma real (barras) con la forma normal (línea).

Picos más altos o colas más gordas en las barras indican retornos concentrados o riesgos de eventos extremos.

Qué indica un buen ajuste:

Si las barras siguen aproximadamente la campana, tu distribución de retornos no tiene colas excesivas.

Señales de alerta:

Colas más gruesas en las barras (más frecuencia de grandes pérdidas/ganancias) que la campana, sugiere riesgo de episodios extremos.

"""


st.markdown(glosario)
