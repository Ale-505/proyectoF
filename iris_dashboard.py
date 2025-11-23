import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Clasificación de Especies de Iris", layout="wide", page_icon="🌸")

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Cargar y preparar datos
@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo'])
    df['especie'] = iris.target
    df['nombre_especie'] = df['especie'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris

@st.cache_resource
def entrenar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    modelo.fit(X_train, y_train)
    return modelo

# Título principal
st.markdown('<p class="main-header">🌸 Dashboard de Clasificación de Especies de Iris</p>', unsafe_allow_html=True)
st.markdown("**Universidad de la Costa - Proyecto Final de Minería de Datos**")
st.markdown("---")

# Cargar datos
df, iris = cargar_datos()

# Barra lateral
st.sidebar.title("📊 Navegación")
pagina = st.sidebar.radio("Seleccionar Página", ["🏠 Inicio", "📈 Exploración de Datos", "🤖 Rendimiento del Modelo", "🔮 Hacer Predicciones"])

# Preparar datos para modelado
X = df.iloc[:, :4]
y = df['especie']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Estandarizar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
modelo = entrenar_modelo(X_train_scaled, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test_scaled)

# Calcular métricas
exactitud = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# PÁGINA DE INICIO
if pagina == "🏠 Inicio":
    st.markdown('<p class="sub-header">Descripción del Proyecto</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Acerca del Dataset de Iris
        El dataset de Iris contiene 150 muestras de flores de iris de tres especies:
        - **Iris Setosa** 🌺
        - **Iris Versicolor** 🌸
        - **Iris Virginica** 🌼
        
        Cada muestra tiene cuatro características:
        1. Longitud del Sépalo (cm)
        2. Ancho del Sépalo (cm)
        3. Longitud del Pétalo (cm)
        4. Ancho del Pétalo (cm)
        
        ### Flujo de Trabajo de la Metodología
        1. **Comprensión de Datos**: Análisis exploratorio de características y distribuciones
        2. **Preprocesamiento de Datos**: Escalado de características usando StandardScaler
        3. **Selección del Modelo**: Clasificador Random Forest (100 estimadores)
        4. **Entrenamiento del Modelo**: División 80-20 entrenamiento-prueba con estratificación
        5. **Evaluación**: Métricas de rendimiento y validación
        6. **Despliegue**: Interfaz interactiva de predicción
        
        ### Justificación del Modelo
        **¿Por qué Random Forest?**
        - Maneja relaciones no lineales entre características
        - Robusto ante valores atípicos
        - Proporciona importancia de características
        - Excelente rendimiento en datasets pequeños y medianos
        - Bajo riesgo de sobreajuste con configuración adecuada
        """)
    
    with col2:
        st.info("### Estadísticas Rápidas")
        st.metric("Total de Muestras", len(df))
        st.metric("Características", 4)
        st.metric("Clases", 3)
        st.metric("Exactitud del Modelo", f"{exactitud:.2%}")

# PÁGINA DE EXPLORACIÓN DE DATOS
elif pagina == "📈 Exploración de Datos":
    st.markdown('<p class="sub-header">Exploración de Datos</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Resumen del Dataset", "📉 Distribuciones", "🔗 Correlaciones"])
    
    with tab1:
        st.markdown("### Muestra del Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Resumen Estadístico")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("### Distribución de Clases")
        col1, col2 = st.columns(2)
        with col1:
            conteo_clases = df['nombre_especie'].value_counts()
            fig = px.bar(x=conteo_clases.index, y=conteo_clases.values, 
                        labels={'x': 'Especie', 'y': 'Cantidad'},
                        title='Distribución de Especies',
                        color=conteo_clases.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(values=conteo_clases.values, names=conteo_clases.index,
                        title='Proporción de Especies')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Distribuciones de Características por Especie")
        caracteristica = st.selectbox("Seleccionar Característica", df.columns[:4])
        
        fig = px.histogram(df, x=caracteristica, color='nombre_especie',
                          marginal='box',
                          title=f'Distribución de {caracteristica}',
                          barmode='overlay',
                          opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos de caja
        fig = px.box(df, x='nombre_especie', y=caracteristica,
                    color='nombre_especie',
                    title=f'{caracteristica} por Especie')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Correlaciones entre Características")
        
        # Matriz de correlación
        matriz_corr = df.iloc[:, :4].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Matriz de Correlación')
        st.pyplot(fig)
        
        # Matriz de dispersión
        st.markdown("### Relaciones entre Pares de Características")
        fig = px.scatter_matrix(df, dimensions=df.columns[:4],
                               color='nombre_especie',
                               title='Matriz de Dispersión de Características')
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# PÁGINA DE RENDIMIENTO DEL MODELO
elif pagina == "🤖 Rendimiento del Modelo":
    st.markdown('<p class="sub-header">Rendimiento del Modelo</p>', unsafe_allow_html=True)
    
    # Mostrar métricas
    st.markdown("### Métricas de Rendimiento")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Exactitud (Accuracy)", f"{exactitud:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precisión (Precision)", f"{precision:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", f"{recall:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1-Score", f"{f1:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                   yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                   ax=ax)
        ax.set_ylabel('Etiqueta Real')
        ax.set_xlabel('Etiqueta Predicha')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Importancia de Características")
        importancia_caracteristicas = pd.DataFrame({
            'caracteristica': df.columns[:4],
            'importancia': modelo.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        fig = px.bar(importancia_caracteristicas, x='importancia', y='caracteristica',
                    orientation='h',
                    title='Importancia de Características en Random Forest',
                    labels={'importancia': 'Importancia', 'caracteristica': 'Característica'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Detalles del Modelo")
        st.info(f"""
        - **Algoritmo**: Clasificador Random Forest
        - **Número de Árboles**: 100
        - **Profundidad Máxima**: 5
        - **Muestras de Entrenamiento**: {len(X_train)}
        - **Muestras de Prueba**: {len(X_test)}
        - **Estado Aleatorio**: 42
        """)
        
        st.markdown("### Interpretación de Métricas")
        st.success("""
        **Exactitud (Accuracy)**: Proporción de predicciones correctas sobre el total.
        
        **Precisión (Precision)**: De todas las predicciones positivas, cuántas fueron correctas.
        
        **Recall (Sensibilidad)**: De todos los casos positivos reales, cuántos fueron detectados.
        
        **F1-Score**: Media armónica entre precisión y recall, útil cuando hay desbalance de clases.
        """)

# PÁGINA DE PREDICCIÓN
elif pagina == "🔮 Hacer Predicciones":
    st.markdown('<p class="sub-header">Predicción Interactiva</p>', unsafe_allow_html=True)
    
    st.markdown("### Ingresa las Medidas de la Flor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitud_sepalo = st.slider("Longitud del Sépalo (cm)", 
                                     float(df.iloc[:, 0].min()), 
                                     float(df.iloc[:, 0].max()), 
                                     float(df.iloc[:, 0].mean()))
        ancho_sepalo = st.slider("Ancho del Sépalo (cm)", 
                                 float(df.iloc[:, 1].min()), 
                                 float(df.iloc[:, 1].max()), 
                                 float(df.iloc[:, 1].mean()))
    
    with col2:
        longitud_petalo = st.slider("Longitud del Pétalo (cm)", 
                                    float(df.iloc[:, 2].min()), 
                                    float(df.iloc[:, 2].max()), 
                                    float(df.iloc[:, 2].mean()))
        ancho_petalo = st.slider("Ancho del Pétalo (cm)", 
                                 float(df.iloc[:, 3].min()), 
                                 float(df.iloc[:, 3].max()), 
                                 float(df.iloc[:, 3].mean()))
    
    # Hacer predicción
    datos_entrada = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
    entrada_escalada = scaler.transform(datos_entrada)
    prediccion = modelo.predict(entrada_escalada)[0]
    probabilidades_prediccion = modelo.predict_proba(entrada_escalada)[0]
    
    mapa_especies = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    especie_predicha = mapa_especies[prediccion]
    
    # Mostrar predicción
    st.markdown("---")
    st.markdown("### Resultado de la Predicción")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.success(f"## 🌸 {especie_predicha}")
        st.markdown("### Niveles de Confianza")
        for i, especie in mapa_especies.items():
            st.progress(float(probabilidades_prediccion[i]), text=f"{especie}: {probabilidades_prediccion[i]:.2%}")
        
        st.markdown("### Datos Ingresados")
        st.info(f"""
        **Longitud Sépalo**: {longitud_sepalo:.2f} cm  
        **Ancho Sépalo**: {ancho_sepalo:.2f} cm  
        **Longitud Pétalo**: {longitud_petalo:.2f} cm  
        **Ancho Pétalo**: {ancho_petalo:.2f} cm
        """)
    
    with col2:
        # Gráfico de dispersión 3D
        st.markdown("### Visualización 3D")
        fig = go.Figure()
        
        # Graficar datos existentes
        for idx_especie, nombre_especie in mapa_especies.items():
            mascara = df['especie'] == idx_especie
            fig.add_trace(go.Scatter3d(
                x=df[mascara].iloc[:, 2],
                y=df[mascara].iloc[:, 3],
                z=df[mascara].iloc[:, 0],
                mode='markers',
                name=nombre_especie,
                marker=dict(size=5, opacity=0.6)
            ))
        
        # Graficar nueva predicción
        fig.add_trace(go.Scatter3d(
            x=[longitud_petalo],
            y=[ancho_petalo],
            z=[longitud_sepalo],
            mode='markers',
            name='Tu Predicción',
            marker=dict(size=15, color='red', symbol='diamond',
                       line=dict(color='black', width=2))
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Longitud Pétalo',
                yaxis_title='Ancho Pétalo',
                zaxis_title='Longitud Sépalo'
            ),
            height=500,
            title='Posición de tu muestra en el espacio de características'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Interpretación")
        st.info("""
        El gráfico 3D muestra la posición de tu muestra (rombo rojo) en relación con 
        todas las muestras del dataset. Las muestras más cercanas a tu predicción 
        pertenecen a la especie predicha, lo que valida el resultado del modelo.
        """)

# Pie de página
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Universidad de la Costa - Curso de Minería de Datos</p>
    <p>Desarrollado por: [Nombres de los Miembros del Equipo]</p>
    <p><i>"Las tres principales virtudes de un programador son: Pereza, Impaciencia y Arrogancia." - Larry Wall</i></p>
</div>
""", unsafe_allow_html=True)