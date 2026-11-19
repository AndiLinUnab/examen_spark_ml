"""App de Streamlit: predicción de deserción / éxito académico con un PipelineModel de Spark ML."""
import json
import os

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
AUTOR = "TU NOMBRE COMPLETO"          # <-- CAMBIAR
BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE, "modelo_final")
RUTA_BANNER = os.path.join(BASE, "banner.png")

st.set_page_config(page_title="Alerta temprana de deserción", page_icon="🎓", layout="wide")

CLASES = {
    "Dropout":  {"es": "Deserción",   "emoji": "🚨", "color": "#C0392B",
                 "msg": "Riesgo alto de abandonar los estudios. Conviene activar acompañamiento académico y financiero."},
    "Enrolled": {"es": "Matriculado", "emoji": "📘", "color": "#D68910",
                 "msg": "Continúa estudiando, pero aún sin perfil claro de graduación. Requiere seguimiento."},
    "Graduate": {"es": "Graduado",    "emoji": "🎓", "color": "#1E8449",
                 "msg": "Va en camino de graduarse. Mantener el seguimiento habitual."},
}

# Estilo propio y coherente: azul marino + turquesa, tipografía Poppins
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }
h1, h2, h3 { color: #0B3C5D; font-weight: 700; }
.stApp { background: #F4F8FB; }
div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid #D6E4EE; border-radius: 14px; padding: 1.2rem; }
.stButton button, div[data-testid="stFormSubmitButton"] button {
    background: #0B3C5D; color: white; border-radius: 10px; font-weight: 600; padding: .6rem 2rem; border: 0; }
.stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover { background: #17A2B8; color: white; }
.resultado { border-radius: 16px; padding: 1.4rem 1.8rem; color: white; margin: .5rem 0 1rem 0; }
.resultado h2 { color: white; margin: 0; }
.resultado p { margin: .4rem 0 0 0; font-size: 1.05rem; }
.pie { text-align: center; color: #5D6D7E; border-top: 1px solid #D6E4EE; margin-top: 2.5rem; padding-top: 1rem; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Recursos en memoria: Spark + modelo (se cargan una sola vez)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Iniciando Spark y cargando el modelo… (la primera vez tarda un poco)")
def cargar_recursos():
    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .appName("dropout_streamlit")
             .master("local[2]")
             .config("spark.driver.memory", "1g")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.constraintPropagation.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    modelo = PipelineModel.load(RUTA_MODELO)
    with open(os.path.join(RUTA_MODELO, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(RUTA_MODELO, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)          # mismo orden que el StringIndexer del entrenamiento
    return spark, modelo, meta, labels


def calcular_derivadas(f):
    """Mismas fórmulas que en el notebook (tarea 3)."""
    e1, e2 = f["curricular_units_1st_sem_enrolled"], f["curricular_units_2nd_sem_enrolled"]
    a1, a2 = f["curricular_units_1st_sem_approved"], f["curricular_units_2nd_sem_approved"]
    g1, g2 = f["curricular_units_1st_sem_grade"], f["curricular_units_2nd_sem_grade"]
    t1 = a1 / e1 if e1 > 0 else 0.0
    t2 = a2 / e2 if e2 > 0 else 0.0
    return {
        "tasa_aprob_1s": t1,
        "tasa_aprob_2s": t2,
        "tasa_aprob_total": (a1 + a2) / (e1 + e2) if (e1 + e2) > 0 else 0.0,
        "variacion_nota": g2 - g1,
        "variacion_tasa_aprob": t2 - t1,
    }


def grupo(col):
    if "1st_sem" in col:
        return "📗 1.er semestre"
    if "2nd_sem" in col:
        return "📙 2.º semestre"
    if col.startswith(("mother", "father")):
        return "👪 Familia"
    if col in ("unemployment_rate", "inflation_rate", "gdp"):
        return "📈 Contexto económico"
    if col in ("debtor", "tuition_fees_up_to_date", "scholarship_holder"):
        return "💳 Situación financiera"
    if col in ("application_mode", "application_order", "course", "daytime_evening_attendance",
               "previous_qualification", "previous_qualification_grade", "admission_grade"):
        return "🏫 Ingreso y estudios previos"
    return "🧑 Datos personales"


# ----------------------------------------------------------------------------
# Encabezado (título + imagen) e instrucciones
# ----------------------------------------------------------------------------
if os.path.exists(RUTA_BANNER):
    st.image(RUTA_BANNER, width="stretch")
st.title("🎓 Alerta temprana de deserción estudiantil")
st.caption("Clasificación multiclase con Spark ML · Dropout · Enrolled · Graduate")

with st.expander("ℹ️ Instrucciones de uso", expanded=True):
    st.markdown("""
1. Complete el formulario con los datos del estudiante. Está organizado en pestañas (datos personales, familia, ingreso, finanzas, semestres y contexto económico).
2. Las variables con **códigos** (curso, ocupación, estado civil…) se eligen de una lista; las numéricas se escriben o se ajustan con los botones **+ / −**.
3. Pulse **Predecir**. La app calcula las variables derivadas (tasas de aprobación y variación de notas) y aplica el modelo.
4. Verá la **clase predicha** destacada y la **probabilidad de cada clase**.
    """)

try:
    spark, modelo, meta, LABELS = cargar_recursos()
except Exception as e:  # mensaje claro si falta la carpeta del modelo
    st.error(f"No se pudo cargar el modelo desde `{RUTA_MODELO}`. Detalle: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Formulario con todas las variables del modelo
# ----------------------------------------------------------------------------
columnas = list(meta["cat_cols"].keys()) + list(meta["num_cols"].keys())
grupos = {}
for c in columnas:
    grupos.setdefault(grupo(c), []).append(c)
orden_tabs = ["🧑 Datos personales", "👪 Familia", "🏫 Ingreso y estudios previos", "💳 Situación financiera",
              "📗 1.er semestre", "📙 2.º semestre", "📈 Contexto económico"]
orden_tabs = [g for g in orden_tabs if g in grupos]

st.subheader("📝 Datos del estudiante")
valores = {}
with st.form("form_estudiante"):
    tabs = st.tabs(orden_tabs)
    for tab, g in zip(tabs, orden_tabs):
        with tab:
            cols_ui = st.columns(3)
            for i, c in enumerate(grupos[g]):
                etiqueta = c.replace("_", " ").capitalize()
                with cols_ui[i % 3]:
                    if c in meta["cat_cols"]:
                        info = meta["cat_cols"][c]
                        ops = info["valores"]
                        valores[c] = st.selectbox(etiqueta, ops, index=ops.index(info["moda"]), key=c)
                    else:
                        info = meta["num_cols"][c]
                        if info["entero"]:
                            valores[c] = float(st.number_input(
                                etiqueta, min_value=int(info["min"]), max_value=int(info["max"]),
                                value=int(round(info["mediana"])), step=1, key=c))
                        else:
                            valores[c] = float(st.number_input(
                                etiqueta, min_value=float(info["min"]), max_value=float(info["max"]),
                                value=float(info["mediana"]), step=0.1, key=c))
    enviado = st.form_submit_button("Predecir")

# ----------------------------------------------------------------------------
# Predicción con el modelo de Spark
# ----------------------------------------------------------------------------
if enviado:
    fila = dict(valores)
    fila.update(calcular_derivadas(fila))                                # (c) variables derivadas
    pdf = pd.DataFrame([fila])
    sdf = spark.createDataFrame(pdf)                                     # (d) DataFrame de Spark
    res = modelo.transform(sdf).select("prediction", "probability").first()

    idx = int(res["prediction"])
    clase = LABELS[idx]                                                  # (e) mismo orden del StringIndexer
    probs = {LABELS[i]: float(p) for i, p in enumerate(res["probability"].toArray())}
    cfg = CLASES.get(clase, {"es": clase, "emoji": "❔", "color": "#0B3C5D", "msg": ""})

    st.subheader("🔎 Resultado")
    st.markdown(f"""
<div class="resultado" style="background:{cfg['color']}">
  <h2>{cfg['emoji']} Clase predicha: {cfg['es']} ({clase})</h2>
  <p>{cfg['msg']}</p>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Probabilidad por clase**")
        tabla = pd.DataFrame({
            "Clase": [f"{CLASES.get(k, {'es': k})['es']} ({k})" for k in probs],
            "Probabilidad": [f"{v * 100:.1f} %" for v in probs.values()],
        })
        st.dataframe(tabla, hide_index=True, width="stretch")
        m_cols = st.columns(len(probs))
        for col, (k, v) in zip(m_cols, probs.items()):
            col.metric(CLASES.get(k, {"es": k})["es"], f"{v * 100:.1f} %")
    with c2:
        st.bar_chart(pd.Series({CLASES.get(k, {"es": k})["es"]: v for k, v in probs.items()}, name="Probabilidad"))

    with st.expander("Ver variables derivadas calculadas"):
        st.dataframe(pd.DataFrame([calcular_derivadas(valores)]).T.rename(columns={0: "valor"}))

# ----------------------------------------------------------------------------
# Pie de página
# ----------------------------------------------------------------------------
st.markdown(f"""<div class="pie">Big Data · Spark ML — Examen, Parte 2<br>
Desarrollado por <b>{AUTOR}</b></div>""", unsafe_allow_html=True)
