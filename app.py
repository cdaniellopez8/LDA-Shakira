import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unicodedata

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="LDA Shakira", page_icon="🎶", layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
def normalizar(s: str):
    if isinstance(s, str):
        s = s.strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        return s
    return s

@st.cache_data
def leer_csv_seguro(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        # último intento, cp1252 típico de Windows
        return pd.read_csv(path, encoding="cp1252")

@st.cache_data
def load_data():
    topic_words = leer_csv_seguro("shakira_lda_topic_words.csv")
    topicos_canciones = leer_csv_seguro("shak_topicos_canciones.csv")
    canciones = pd.read_excel("shak.xlsx", engine="openpyxl")

    # --- aseguramos tipos/columnas esperadas ---
    req_tw = {"topic","word","weight","rank"}
    req_tc = {"song","year","dominant_topic","nombre_topic","dominant_prob",
              "Topic_1","Topic_2","Topic_3","Topic_4","Topic_5"}
    req_shak = {"song","year","lyrics"}

    if not req_tw.issubset(topic_words.columns):
        raise ValueError(f"shakira_lda_topic_words.csv debe tener columnas {sorted(req_tw)}")
    if not req_tc.issubset(topicos_canciones.columns):
        raise ValueError(f"shak_topicos_canciones.csv debe tener columnas {sorted(req_tc)}")
    if not req_shak.issubset(canciones.columns):
        raise ValueError(f"shak.xlsx debe tener columnas {sorted(req_shak)}")

    # tipos
    topic_words["topic"] = topic_words["topic"].astype(int)
    for col in ["weight","rank"]:
        topic_words[col] = pd.to_numeric(topic_words[col], errors="coerce")

    topicos_canciones["year"] = pd.to_numeric(topicos_canciones["year"], errors="coerce").astype("Int64")
    topicos_canciones["dominant_topic"] = pd.to_numeric(topicos_canciones["dominant_topic"], errors="coerce").astype(int)
    for i in range(1,6):
        topicos_canciones[f"Topic_{i}"] = pd.to_numeric(topicos_canciones[f"Topic_{i}"], errors="coerce")

    canciones["year"] = pd.to_numeric(canciones["year"], errors="coerce").astype("Int64")

    # normalización para join robusto
    canciones["song_norm"] = canciones["song"].apply(normalizar)
    topicos_canciones["song_norm"] = topicos_canciones["song"].apply(normalizar)

    # mapa id->nombre de tópico
    topic_name_map = (
        topicos_canciones[["dominant_topic","nombre_topic"]]
        .dropna()
        .drop_duplicates()
        .set_index("dominant_topic")["nombre_topic"]
        .to_dict()
    )

    return topic_words, topicos_canciones, canciones, topic_name_map

topic_words, topicos_canciones, canciones, topic_name_map = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎶 LDA – Shakira")
modo = st.sidebar.radio("Modo de exploración", ["Por Canción", "Por Tópico", "Vista Global"])

# -----------------------------
# POR CANCIÓN
# -----------------------------
if modo == "Por Canción":
    cancion_sel = st.sidebar.selectbox("Selecciona una canción", topicos_canciones["song"].tolist())
    cancion_norm = normalizar(cancion_sel)

    fila = topicos_canciones[topicos_canciones["song_norm"] == cancion_norm]
    if fila.empty:
        st.error("No se encontró la canción seleccionada en shak_topicos_canciones.csv")
        st.stop()
    fila = fila.iloc[0]

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"### 📌 {cancion_sel} ({int(fila['year']) if pd.notna(fila['year']) else 's/f'})")
        st.markdown(f"**Tópico dominante:** {fila['nombre_topic']}  &nbsp;&nbsp;|&nbsp;&nbsp; **Prob:** {fila['dominant_prob']:.3f}")
    with col2:
        # palabras clave del tópico dominante
        top_id = int(fila["dominant_topic"])
        palabras = (topic_words[topic_words["topic"] == top_id]
                    .sort_values("weight", ascending=False)
                    .head(12))[["word","weight"]]
        st.markdown("**Top palabras del tópico dominante**")
        st.dataframe(palabras, use_container_width=True, hide_index=True)

    # letra
    letra_row = canciones[canciones["song_norm"] == cancion_norm]
    with st.expander("📖 Ver letra"):
        if letra_row.empty:
            st.warning("No se encontró la letra en shak.xlsx")
        else:
            st.write(letra_row["lyrics"].values[0])

   
    # -----------------------------
    # DISTRIBUCIÓN DE TÓPICOS EN LA CANCIÓN
    # -----------------------------
    st.markdown("### 🎶 Distribución de tópicos en la canción seleccionada")

    # aseguramos que cancion no esté vacío
    if "cancion" in locals():
        selected_song = cancion
    else:
        selected_song = st.session_state.get("cancion", "Canción desconocida")

    row = topicos_canciones[topicos_canciones["song_norm"] == cancion_norm].iloc[0]
    topic_values = {topic_name_map.get(i, f"Topic {i}"): row[f"Topic_{i}"] for i in range(1, 6)}

    df_song_topics = pd.DataFrame({
        "Tópico": list(topic_values.keys()),
        "Probabilidad": list(topic_values.values())
    })

    fig_bar = px.bar(
        df_song_topics, 
        x="Tópico", 
        y="Probabilidad", 
        title=f"Distribución de tópicos en '{selected_song}'",  # ✅ ahora siempre existe
        text="Probabilidad"
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition="outside")
    fig_bar.update_layout(yaxis=dict(range=[0,1]))
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# POR TÓPICO
# -----------------------------
elif modo == "Por Tópico":
    top_id = st.sidebar.selectbox("Selecciona un tópico (ID)", sorted(topic_words["topic"].unique()))
    nombre = topic_name_map.get(int(top_id), f"Tópico {top_id}")
    st.markdown(f"### 📌 Tópico {top_id}: {nombre}")

    # palabras del tópico
    n_pal = st.sidebar.slider("N° de palabras clave a mostrar", 5, 30, 15, 1)
    palabras = (topic_words[topic_words["topic"] == int(top_id)]
                .sort_values("weight", ascending=False)
                .head(n_pal))[["word","weight"]]
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("**Palabras más importantes**")
        st.dataframe(palabras, use_container_width=True, hide_index=True)
    with col2:
        # nube de barras de palabras
        fig_words = px.bar(palabras.sort_values("weight"),
                           x="weight", y="word", orientation="h",
                           labels={"weight":"Peso","word":"Palabra"},
                           title="Importancia de palabras")
        st.plotly_chart(fig_words, use_container_width=True)

    # canciones donde este tópico es dominante
    st.markdown("#### 🎵 Canciones donde este tópico es dominante")
    canciones_top = (topicos_canciones[topicos_canciones["dominant_topic"] == int(top_id)]
                     .sort_values(["year","dominant_prob"], ascending=[True, False])
                     [["song","year","dominant_prob"]])
    st.dataframe(canciones_top, use_container_width=True, hide_index=True)

    # evolución del tópico (promedio del Topic_k por año en TODAS las canciones)
    st.markdown("#### 📈 Evolución del tópico por año (promedio en todas las canciones)")
    col_topic = f"Topic_{int(top_id)}"
    df_topic_year = (topicos_canciones.groupby("year", dropna=True)[col_topic]
                     .mean()
                     .reset_index()
                     .sort_values("year"))
    fig_line_one = px.line(df_topic_year, x="year", y=col_topic, markers=True,
                           labels={col_topic:"Prob. promedio", "year":"Año"},
                           title=f"Evolución del Tópico {top_id} – {nombre}")
    fig_line_one.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line_one, use_container_width=True)

# -----------------------------
# VISTA GLOBAL (5 líneas con nombres reales)
# -----------------------------
else:
    st.markdown("### 📊 Evolución de **todos** los tópicos por año")
    df_all = (topicos_canciones
              .groupby("year", dropna=True)[[f"Topic_{i}" for i in range(1,6)]]
              .mean()
              .reset_index()
              .sort_values("year"))

    fig_all = go.Figure()
    for i in range(1, 6):
        nombre = topic_name_map.get(i, f"Topic {i}")  # usa el nombre si existe
        fig_all.add_trace(go.Scatter(
            x=df_all["year"], y=df_all[f"Topic_{i}"],
            mode="lines+markers", name=nombre
        ))

    fig_all.update_layout(
        title="Evolución de los 5 tópicos (media por año)",
        xaxis_title="Año",
        yaxis_title="Probabilidad promedio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Mapa rápido de id -> nombre
    st.markdown("**Mapa de nombres de tópicos**")
    df_names = pd.DataFrame(
        [{"topic_id": k, "nombre_topic": v} for k, v in topic_name_map.items()]
    ).sort_values("topic_id")
    st.dataframe(df_names, hide_index=True, use_container_width=True)

# Footer bonito
st.markdown(
    "<hr><div style='text-align:center;color:gray;font-size:0.9em;'>"
    "Presentado por Carlos D. Lopez Perez | LDA de canciones de Shakira"
    "</div>",
    unsafe_allow_html=True
)