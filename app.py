import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Configuración de la página
st.set_page_config(
    page_title="Mental Health & Productivity Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_dataset.csv")
    return df

df = load_data()

# Sidebar de navegación
st.sidebar.title("🧠 Mental Health Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Análisis Exploratorio", "Análisis Predictivo", "Orquestación de Datos"]
)

# ==================== DASHBOARD ====================
if page == "Dashboard":
    st.title("📊 Dashboard General")
    st.markdown("---")

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Empleados", len(df))

    with col2:
        avg_stress = df["Stress_Level"].mean()
        st.metric("Estrés Promedio", f"{avg_stress:.1f}")

    with col3:
        avg_sleep = df["Sleep_Hours"].mean()
        st.metric("Horas Sueño Promedio", f"{avg_sleep:.1f}")

    with col4:
        avg_productivity = df["Productivity_Score"].mean()
        st.metric("Productividad Promedio", f"{avg_productivity:.1f}")

    st.markdown("---")

    # Filtros interactivos
    st.subheader("🔍 Filtros Interactivos")

    col1, col2, col3 = st.columns(3)

    with col1:
        stress_filter = st.multiselect(
            "Nivel de Estrés",
            options=sorted(df["Stress_Level"].unique()),
            default=sorted(df["Stress_Level"].unique())
        )

    with col2:
        productivity_filter = st.multiselect(
            "Nivel de Productividad",
            options=df["productivity_level"].unique(),
            default=df["productivity_level"].unique()
        )

    with col3:
        burnout_filter = st.multiselect(
            "Riesgo de Burnout",
            options=df["Burnout_Risk"].unique(),
            default=df["Burnout_Risk"].unique()
        )

    # Aplicar filtros
    filtered_df = df[
        (df["Stress_Level"].isin(stress_filter)) &
        (df["productivity_level"].isin(productivity_filter)) &
        (df["Burnout_Risk"].isin(burnout_filter))
    ]

    st.markdown(f"**Registros filtrados:** {len(filtered_df)}")

    # Tabla de datos
    st.subheader("📋 Tabla de Datos")
    st.dataframe(filtered_df, use_container_width=True)

    # Visualizaciones principales
    col1, col2 = st.columns(2)

    with col1:
        # Histograma de Estrés
        fig_stress = px.histogram(
            filtered_df,
            x="Stress_Level",
            nbins=10,
            title="Distribución de Niveles de Estrés",
            color_discrete_sequence=["#FF6B6B"]
        )
        fig_stress.update_layout(showlegend=False)
        st.plotly_chart(fig_stress, use_container_width=True)

        # Scatter: Sleep vs Productivity
        fig_sleep_prod = px.scatter(
            filtered_df,
            x="Sleep_Hours",
            y="Productivity_Score",
            color="Stress_Level",
            title="Horas de Sueño vs Productividad",
            color_continuous_scale="RdYlBu_r"
        )
        st.plotly_chart(fig_sleep_prod, use_container_width=True)

    with col2:
        # Histograma de Productividad
        fig_prod = px.histogram(
            filtered_df,
            x="Productivity_Score",
            nbins=20,
            title="Distribución de Scores de Productividad",
            color="productivity_level",
            color_discrete_map={"Baja": "#FF9999", "Media": "#FFD700", "Alta": "#90EE90"}
        )
        st.plotly_chart(fig_prod, use_container_width=True)

        # Boxplot Stress por Burnout
        fig_box = px.box(
            filtered_df,
            x="Burnout_Risk",
            y="Stress_Level",
            color="Burnout_Risk",
            title="Nivel de Estrés por Riesgo de Burnout"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Matriz de correlación
    st.subheader("🔗 Matriz de Correlación")
    numeric_cols = ["Age", "Work_Hours_Per_Week", "Stress_Level", "Sleep_Hours",
                    "Productivity_Score", "Physical_Activity_Hours"]

    corr_matrix = filtered_df[numeric_cols].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig_corr.update_layout(title="Correlación entre Variables Numéricas")
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== ANÁLISIS EXPLORATORIO ====================
elif page == "Análisis Exploratorio":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    st.markdown("---")

    # Vista general
    st.subheader("📊 Información General del Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Filas", df.shape[0])
        st.metric("Columnas", df.shape[1])

    with col2:
        st.write("**Variables:**")
        st.write(", ".join(df.columns.tolist()))

    # Estadísticas descriptivas
    st.subheader("📈 Estadísticas Descriptivas")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    st.dataframe(numeric_df.describe().round(2), use_container_width=True)

    # Distribuciones
    st.subheader("📉 Distribuciones de Variables Numéricas")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig_age = px.histogram(df, x="Age", nbins=20, title="Edad")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_work = px.histogram(df, x="Work_Hours_Per_Week", nbins=15, title="Horas Trabajadas por Semana")
        st.plotly_chart(fig_work, use_container_width=True)

    with col3:
        fig_sleep = px.histogram(df, x="Sleep_Hours", nbins=15, title="Horas de Sueño")
        st.plotly_chart(fig_sleep, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig_stress = px.histogram(df, x="Stress_Level", nbins=10, title="Nivel de Estrés")
        st.plotly_chart(fig_stress, use_container_width=True)

    with col2:
        fig_prod = px.histogram(df, x="Productivity_Score", nbins=20, title="Score de Productividad")
        st.plotly_chart(fig_prod, use_container_width=True)

    with col3:
        fig_activity = px.histogram(df, x="Physical_Activity_Hours", nbins=15, title="Actividad Física (horas)")
        st.plotly_chart(fig_activity, use_container_width=True)

    # Análisis categórico
    st.subheader("📊 Distribución de Variables Categóricas")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_counts = df["Gender"].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title="Distribución por Género")
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        burnout_counts = df["Burnout_Risk"].value_counts()
        fig_burnout = px.pie(values=burnout_counts.values, names=burnout_counts.index, title="Riesgo de Burnout")
        st.plotly_chart(fig_burnout, use_container_width=True)

    with col3:
        support_counts = df["Mental_Health_Support_Access"].value_counts()
        fig_support = px.pie(values=support_counts.values, names=support_counts.index, title="Acceso a Soporte Mental")
        st.plotly_chart(fig_support, use_container_width=True)

    # Análisis cruzado
    st.subheader("📈 Análisis Cruzado: Estrés vs Productividad")
    cross_tab = pd.crosstab(df["Stress_Level"], df["productivity_level"], normalize="index") * 100
    st.dataframe(cross_tab.round(2), use_container_width=True)

    # Work mode analysis
    st.subheader("💼 Análisis por Modalidad de Trabajo")
    work_mode_stats = df.groupby("Work_Mode")[["Stress_Level", "Productivity_Score", "Sleep_Hours"]].mean().round(2)
    st.dataframe(work_mode_stats, use_container_width=True)

# ==================== ANÁLISIS PREDICTIVO ====================
elif page == "Análisis Predictivo":
    st.title("🔮 Análisis Predictivo")
    st.markdown("---")

    st.info("""
    Esta sección presenta correlaciones y patrones identificados que pueden servir
    como base para modelos predictivos de productividad y salud mental.
    """)

    # Correlación con productividad
    numeric_cols = ["Age", "Work_Hours_Per_Week", "Stress_Level", "Sleep_Hours",
                    "Productivity_Score", "Physical_Activity_Hours"]
    corr_with_productivity = df[numeric_cols].corr()["Productivity_Score"].sort_values()

    st.subheader("📊 Correlación con Productividad")
    fig_corr_prod = px.bar(
        x=corr_with_productivity.values,
        y=corr_with_productivity.index,
        orientation='h',
        title="Correlación de Variables con Productividad",
        color=corr_with_productivity.values,
        color_continuous_scale="RdYlBu",
        range_color=[-1, 1]
    )
    fig_corr_prod.update_layout(yaxis_title="Variable", xaxis_title="Correlación")
    st.plotly_chart(fig_corr_prod, use_container_width=True)

    # Insights
    st.subheader("💡 Insights Principales")

    col1, col2 = st.columns(2)

    with col1:
        st.success("** Factores Positivos para Productividad:**")
        st.write("- Más horas de sueño → mayor productividad")
        st.write("- Mayor actividad física → mejor rendimiento")
        st.write("- Menor nivel de estrés → mayor productividad")

    with col2:
        st.error("** Factores Negativos para Productividad:**")
        st.write("- Más horas de trabajo semanal → menor productividad (rendimientos decrecientes)")
        st.write("- Mayor estrés → scores más bajos")
        st.write("- Burnout risk alto → desempeño reducido")

    # Scatter matrix
    st.subheader("🔬 Matriz de Dispersión")
    st.markdown("Explora relaciones entre múltiples variables simultáneamente.")

    selected_vars = st.multiselect(
        "Selecciona variables para la matriz:",
        numeric_cols,
        default=["Stress_Level", "Sleep_Hours", "Productivity_Score", "Work_Hours_Per_Week"]
    )

    if len(selected_vars) >= 2:
        fig_scatter_matrix = px.scatter_matrix(
            df,
            dimensions=selected_vars,
            color="productivity_level",
            title="Matriz de Dispersión por Nivel de Productividad"
        )
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)

    # Predicción simple basada en reglas
    st.subheader("🎯 Clasificación por Reglas Simples")
    st.markdown("""
    Basado en el análisis, se pueden establecer reglas para predecir productividad:
    - **Alta productividad**: Stress_Level ≤ 4, Sleep_Hours ≥ 7, Productivity_Score ≥ 70
    - **Baja productividad**: Stress_Level ≥ 7, Sleep_Hours ≤ 6, Productivity_Score ≤ 50
    """)

    # Simulación de predicción
    def classify_productivity(row):
        if row["Productivity_Score"] >= 70 and row["Stress_Level"] <= 4 and row["Sleep_Hours"] >= 7:
            return "Alta (predicho)"
        elif row["Productivity_Score"] <= 50 and row["Stress_Level"] >= 7 and row["Sleep_Hours"] <= 6:
            return "Baja (predicho)"
        else:
            return "Media (predicho)"

    df["predicted_level"] = df.apply(classify_productivity, axis=1)

    accuracy = (df["predicted_level"].str.split().str[0] == df["productivity_level"]).mean()
    st.write(f"**Precisión aproximada de reglas simples:** {accuracy:.1%}")

# ==================== ORQUESTACIÓN DE DATOS ====================
elif page == "Orquestación de Datos":
    st.title("🔧 Orquestación de Datos - Pipeline ETL")
    st.markdown("---")

    st.subheader("📋 Arquitectura del Pipeline")

    st.markdown("""
    El proceso ETL (Extract, Transform, Load) implementado sigue la siguiente arquitectura:
    """)

    # Diagrama del pipeline
    st.image("dag.png", caption="Diagrama del Pipeline ETL", use_column_width=True)

    # Explicación detallada
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📥 Extracción")
        st.markdown("""
        - **Fuente**: Dataset desde Kaggle
        - **Archivo**: `mental_health_productivity_2026.csv`
        - **Formato**: CSV
        - **Método**: Descarga mediante `kagglehub`
        - **Tamaño**: 1500 registros, 13 columnas
        """)

    with col2:
        st.markdown("### ⚙️ Transformación")
        st.markdown("""
        - **Limpieza**: Eliminación de valores nulos (`dropna()`)
        - **Validación**: Revisión de tipos de datos
        - **Derivación**: Creación de `productivity_level`
          - Baja: 0-40
          - Media: 41-70
          - Alta: 71-100
        """)

    with col3:
        st.markdown("### 💾 Carga")
        st.markdown("""
        - **Archivo destino**: `cleaned_dataset.csv`
        - **Ubicación**: `./data/`
        - **Formato**: CSV sin índice
        - **Uso**: Input para Streamlit app
        - **Estado**: Listo para análisis
        """)

    # Código del ETL
    st.subheader("📜 Código del Proceso ETL")
    st.code("""
# ET = EXTRACCIÓN + TRANSFORMACIÓN
# L = CARGA

import pandas as pd

# EXTRACCIÓN
df = pd.read_csv("mental_health_productivity_2026.csv")

# TRANSFORMACIÓN
# 1. Eliminar nulos
df = df.dropna()

# 2. Crear variable derivada
df["productivity_level"] = pd.cut(
    df["Productivity_Score"],
    bins=[0, 40, 70, 100],
    labels=["Baja", "Media", "Alta"]
)

# CARGA
df.to_csv("cleaned_dataset.csv", index=False)
    """, language="python")

    # Flujo de datos
    st.subheader("🔄 Flujo de Datos")
    st.markdown("""
    ```
    Kaggle Dataset → Extracción → Limpieza → Feature Engineering → Dataset Limpio → Streamlit App
         .csv              ↓         ↓             ↓                  .csv              ↓
    (raw data)      (1500 x 13)  (sin nulos)  (+1 columna derivada)  (listo)      (visualización)
    ```""")

    # Programación del pipeline (simulación Airflow)
    st.subheader("🛩️ Simulación con Apache Airflow")
    st.markdown("""
    El DAG `mental_health_etl` consta de las siguientes tareas:

    ```python
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime

    def extract_data():
        # Descargar desde Kaggle
        pass

    def clean_data():
        # Eliminar nulos, tipos correctos
        pass

    def feature_engineering():
        # Crear productivity_level
        pass

    def load_data():
        # Guardar cleaned_dataset.csv
        pass

    with DAG('mental_health_etl', start_date=datetime(2024,1,1)) as dag:
        t1 = PythonOperator(task_id='extract_data', python_callable=extract_data)
        t2 = PythonOperator(task_id='clean_data', python_callable=clean_data)
        t3 = PythonOperator(task_id='feature_engineering', python_callable=feature_engineering)
        t4 = PythonOperator(task_id='load_data', python_callable=load_data)

        t1 >> t2 >> t3 >> t4
    ```
    """)

    st.success("El pipeline se ejecuta diariamente para actualizar el dataset limpio.")

    # ==================== EJERCICIOS DE ORQUESTACIÓN (AIRFLOW) ====================
    st.markdown("---")
    st.subheader("✈️ Ejercicios de Orquestación (Airflow)")

    # EJERCICIO 1
    st.markdown("### 📘 Ejercicio 1: Pipeline Secuencial Simple")
    st.markdown("""
    **Objetivo:** Ejecutar tres tareas en secuencia estricta: extraer datos, limpiarlos y cargarlos.
    
    **Aplicación al proyecto:** Este flujo básico representa el proceso mínimo para obtener un dataset 
    utilizable. Primero se extrae el CSV original desde Kaggle, luego se eliminan los valores nulos, 
    y finalmente se guarda el archivo limpio para su posterior análisis en la app Streamlit.
    """)

    st.code("""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_data():
    '''
    Tarea 1: Extraer el dataset desde Kaggle.
    Simula la descarga de mental_health_productivity_2026.csv
    '''
    import kagglehub
    path = kagglehub.dataset_download("shadab80k/mental-health-productivity-2026")
    print(f"Dataset descargado en: {path}")

def clean_data():
    '''
    Tarea 2: Limpiar datos eliminando nulos.
    Lee el archivo descargado y elimina filas con valores faltantes.
    '''
    import pandas as pd
    df = pd.read_csv("mental_health_productivity_2026.csv")
    df_clean = df.dropna()
    print(f"Registros originales: {len(df)}")
    print(f"Registros después de limpieza: {len(df_clean)}")
    df_clean.to_csv("temp_clean.csv", index=False)

def load_data():
    '''
    Tarea 3: Cargar el dataset limpio a la carpeta data/.
    Mueve el archivo procesado a su ubicación final.
    '''
    import shutil
    shutil.move("temp_clean.csv", "data/cleaned_dataset.csv")
    print("Dataset limpio guardado en data/cleaned_dataset.csv")

# Definición del DAG
with DAG(
    dag_id='mental_health_pipeline_simple',
    description='Pipeline ETL secuencial simple',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        doc_md="Extrae dataset desde Kaggle"
    )
    
    t2 = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        doc_md="Limpia valores nulos del dataset"
    )
    
    t3 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        doc_md="Guarda dataset limpio en data/"
    )
    
    #Flujo secuencial: t1 -> t2 -> t3
    t1 >> t2 >> t3
    """, language="python")

    st.info("""
    **Relación con el proyecto:**
    - `extract_data` → Descarga el CSV original (1500 empleados × 13 columnas)
    - `clean_data` → Elimina nulos (en este caso no hay,但 genera el archivo limpio)
    - `load_data` → Genera `cleaned_dataset.csv` que consume `app.py`
    """)

    # EJERCICIO 2
    st.markdown("---")
    st.markdown("### 📘 Ejercicio 2: Pipeline con Feature Engineering")
    st.markdown("""
    **Objetivo:** Incorporar una etapa de transformación avanzada que cree nuevas variables derivadas.
    
    **Aplicación al proyecto:** Este pipeline añade la creación de `productivity_level`, una variable 
    categórica derivada de `Productivity_Score` que clasifica a los empleados en Baja, Media y Alta 
    productividad. Esta variable es clave para el análisis predictivo y los filtros interactivos de la app.
    """)

    st.code("""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

def extract_data():
    '''Extraer dataset desde Kaggle'''
    import kagglehub
    path = kagglehub.dataset_download("shadab80k/mental-health-productivity-2026")
    df = pd.read_csv(path + "/mental_health_productivity_2026.csv")
    df.to_csv("raw_data.csv", index=False)
    print("Extracción completada: raw_data.csv")

def clean_data():
    '''Limpiar datos y validar tipos'''
    df = pd.read_csv("raw_data.csv")
    
    # Eliminar nulos
    df_clean = df.dropna()
    
    # Validar tipos (opcional pero recomendado)
    numeric_cols = ['Age', 'Work_Hours_Per_Week', 'Stress_Level', 
                    'Sleep_Hours', 'Productivity_Score', 'Physical_Activity_Hours']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean.to_csv("clean_data.csv", index=False)
    print("Limpieza completada: clean_data.csv")

def feature_engineering():
    '''
    Crear variable derivada productivity_level.
    Bins: 0-40 → Baja, 41-70 → Media, 71-100 → Alta
    '''
    df = pd.read_csv("clean_data.csv")
    
    # Crear categoría de productividad
    df["productivity_level"] = pd.cut(
        df["Productivity_Score"],
        bins=[0, 40, 70, 100],
        labels=["Baja", "Media", "Alta"]
    ).astype(str)
    
    df.to_csv("features_data.csv", index=False)
    print("Feature engineering completado: productivity_level creada")
    print("Distribución:")
    print(df["productivity_level"].value_counts())

def load_data():
    '''Cargar dataset final a data/'''
    import shutil
    shutil.move("features_data.csv", "data/cleaned_dataset.csv")
    print("Carga completada: data/cleaned_dataset.csv listo para Streamlit")

with DAG(
    dag_id='mental_health_etl_with_features',
    description='Pipeline ETL con ingeniería de características',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )
    
    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data
    )
    
    fe = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )
    
    load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    # Flujo: extract -> clean -> feature_engineering -> load
    extract >> clean >> fe >> load
    """, language="python")

    st.success("""
    **Variable creada:** `productivity_level` (categórica: Baja/Media/Alta)
    
    **Uso en la app:** 
    - Filtros en Dashboard
    - Gráficos por nivel de productividad
    - Análisis predictivo por categoría
    """)

    # EJERCICIO 3
    st.markdown("---")
    st.markdown("### 📘 Ejercicio 3: Pipeline con Tareas Paralelas")
    st.markdown("""
    **Objetivo:** Ejecutar múltiples transformaciones en paralelo para optimizar tiempo de procesamiento.
    
    **Aplicación al proyecto:** Después de la limpieza, se pueden ejecutar en paralelo:
    
    1. **feature_engineering** → Crea `productivity_level` (bins de productividad)
    2. **validation** → Valida rangos de variables (Stress 1-10, Sleep 0-24, etc.)
    3. **data_quality_check** → Verifica consistencia (ej. Work_Hours no > 168/semana)
    
    Ambas tareas deben completarse antes de cargar el dataset final. Esto reduce el tiempo total 
    del pipeline si la validación es costosa (ej. cientos de reglas de negocio).
    """)

    st.code("""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

def extract_data():
    '''Tarea 1: Extraer datos (sin cambios)'''
    import kagglehub
    path = kagglehub.dataset_download("shadab80k/mental-health-productivity-2026")
    df = pd.read_csv(path + "/mental_health_productivity_2026.csv")
    df.to_csv("raw_data.csv", index=False)

def clean_data():
    '''Tarea 2: Limpiar datos (sin cambios)'''
    df = pd.read_csv("raw_data.csv")
    df_clean = df.dropna()
    df_clean.to_csv("clean_data.csv", index=False)

def feature_engineering():
    '''Tarea 3A (paralela): Crear productividad_level'''
    df = pd.read_csv("clean_data.csv")
    df["productivity_level"] = pd.cut(
        df["Productivity_Score"],
        bins=[0, 40, 70, 100],
        labels=["Baja", "Media", "Alta"]
    ).astype(str)
    df.to_csv("with_features.csv", index=False)
    print("Feature engineering completado")

def validation():
    '''Tarea 3B (paralela): Validar rangos de variables clave'''
    df = pd.read_csv("clean_data.csv")
    
    errors = []
    
    # Validaciones
    if (df["Stress_Level"] < 1).any() or (df["Stress_Level"] > 10).any():
        errors.append("Stress_Level fuera de rango [1-10]")
    
    if (df["Sleep_Hours"] < 0).any() or (df["Sleep_Hours"] > 24).any():
        errors.append("Sleep_Hours fuera de rango [0-24]")
    
    if (df["Work_Hours_Per_Week"] > 168).any():
        errors.append("Work_Hours_Per_Week excede 168 hrs/semana")
    
    if (df["Productivity_Score"] < 0).any() or (df["Productivity_Score"] > 100).any():
        errors.append("Productivity_Score fuera de rango [0-100]")
    
    if errors:
        raise ValueError(f"Errores de validación: {errors}")
    else:
        print("✅ Todas las validaciones pasaron")
    
    # Copiar archivo limpio para el merge
    df.to_csv("validated_data.csv", index=False)

def merge_and_load():
    '''Tarea 4: Combinar resultados y cargar (después de Paralelas)'''
    import shutil
    
    # Leer datos con features
    df_fe = pd.read_csv("with_features.csv")
    
    # En un caso real, podríamos hacer merge si validation modifica datos
    # Aquí ambos producen el mismo df limpio, verification es solo checks
    
    # Si validation pasó, usamos el archivo con features
    df_fe.to_csv("data/cleaned_dataset.csv", index=False)
    shutil.copy("validated_data.csv", "data/validation_log.csv")
    
    print("Carga completada: dataset final con features")

with DAG(
    dag_id='mental_health_parallel_pipeline',
    description='Pipeline ETL con tareas paralelas',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    # Tarea 1: Extracción
    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )
    
    # Tarea 2: Limpieza
    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data
    )
    
    # Tarea 3A: Feature Engineering (paralela)
    fe_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )
    
    # Tarea 3B: Validación (paralela a FE)
    validation_task = PythonOperator(
        task_id='validation',
        python_callable=validation
    )
    
    # Tarea 4: Carga (después de AMBAS paralelas)
    load = PythonOperator(
        task_id='load_data',
        python_callable=merge_and_load
    )
    
    # Dependencias:
    # extract >> clean >> [fe_task, validation_task] >> load
    extract >> clean
    clean >> [fe_task, validation_task]  # Paralelo
    fe_task >> load
    validation_task >> load
    """, language="python")

    st.info("""
    ** Beneficios de paralelización:**
    - `feature_engineering` y `validation` se ejecutan simultáneamente
    - Si `validation` tarda 30s y `feature_engineering` tarda 20s, el bloque paralelo 
      toma ~30s (no 50s), reduciendo tiempo total del pipeline
    - Útil cuando se agregan múltiples transformaciones o validaciones complejas
    
    **Aplicación real en proyecto:**
    - `feature_engineering` → Crea `productivity_level`
    - `validation` → Revisa rangos de Stress_Level (1-10), Sleep_Hours (0-24), etc.
    - Ambas deben pasar antes de cargar el dataset final que consume Streamlit
    """)

    st.markdown("---")
    st.subheader("🎯 Resumen de Ejercicios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Ejercicio 1: Secuencial**
        - 3 tareas en línea
        - Más simple, fácil de debuggear
        - Tiempo: T1 + T2 + T3
        """)

    with col2:
        st.markdown("""
        **Ejercicio 2: Con FE**
        - 4 tareas secuenciales
        - Incluye feature engineering
        - Crea `productivity_level`
        """)

    with col3:
        st.markdown("""
        **Ejercicio 3: Paralelo**
        - 4 tareas con branching
        - `clean` → `[FE + validation]` → `load`
        - Optimiza tiempo de ejecución
        """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Proyecto:** Ingeniería de Datos  
**Tema:** Mental Health & Productivity  
**Estudiante:** Mariana Gutierrez
""")
