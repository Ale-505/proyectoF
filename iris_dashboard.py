import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificación de Especies Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title("🌸 Proyecto de Clasificación de Especies Iris")
st.markdown("### Universidad de la Costa - Data Mining")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    # Cargar el CSV (asegúrate de tener el archivo Iris.csv en el mismo directorio)
    df = pd.read_csv('Iris.csv')
    return df

# Cargar y preparar los datos
try:
    df = load_data()
    st.sidebar.success("✅ Datos cargados correctamente")
except:
    st.error("⚠️ No se pudo cargar el archivo Iris.csv. Asegúrate de que esté en el directorio correcto.")
    st.stop()

# Sidebar - Información del equipo
st.sidebar.title("👥 Información del Equipo")
st.sidebar.markdown("""
**Integrantes:**
-ALEJANDRO ESCORCIA
-ASHLEY URUETA

**Profesor:**
José Escorcia-Gutierrez, Ph.D.

**Curso:** Data Mining
""")

# Preparar datos para el modelo
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Codificar las especies
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenar el modelo Random Forest
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Crear tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visualización de Datos",
    "📈 Comprensión de los Datos",
    "🤖 Entrenamiento del Modelo",
    "🎯 Predicciones 3D",
    "📉 Predicciones 2D"
])

# ==================== TAB 1: VISUALIZACIÓN DE DATOS ====================
with tab1:
    st.header("📊 Visualización Exploratoria de los Datos")
    
    # Histogramas
    st.subheader("1. Distribución de Características (Histogramas)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='SepalLengthCm', color='Species', 
                           title='Distribución de Longitud del Sépalo',
                           labels={'SepalLengthCm': 'Longitud del Sépalo (cm)', 'count': 'Frecuencia'},
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.histogram(df, x='PetalLengthCm', color='Species',
                           title='Distribución de Longitud del Pétalo',
                           labels={'PetalLengthCm': 'Longitud del Pétalo (cm)', 'count': 'Frecuencia'},
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.histogram(df, x='SepalWidthCm', color='Species',
                           title='Distribución de Ancho del Sépalo',
                           labels={'SepalWidthCm': 'Ancho del Sépalo (cm)', 'count': 'Frecuencia'},
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.histogram(df, x='PetalWidthCm', color='Species',
                           title='Distribución de Ancho del Pétalo',
                           labels={'PetalWidthCm': 'Ancho del Pétalo (cm)', 'count': 'Frecuencia'},
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # Mapa de calor de correlaciones
    st.subheader("2. Mapa de Calor de Correlaciones")
    corr_matrix = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                         text_auto='.2f',
                         aspect='auto',
                         color_continuous_scale='RdBu_r',
                         title='Matriz de Correlación entre Características',
                         labels=dict(x="Características", y="Características", color="Correlación"))
    fig_corr.update_xaxes(tickangle=45)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Boxplot por especie
    st.subheader("3. Boxplot por Especie")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_box1 = px.box(df, x='Species', y='SepalLengthCm', color='Species',
                         title='Longitud del Sépalo por Especie',
                         labels={'SepalLengthCm': 'Longitud (cm)', 'Species': 'Especie'},
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_box1, use_container_width=True)
        
        fig_box2 = px.box(df, x='Species', y='PetalLengthCm', color='Species',
                         title='Longitud del Pétalo por Especie',
                         labels={'PetalLengthCm': 'Longitud (cm)', 'Species': 'Especie'},
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_box2, use_container_width=True)
    
    with col2:
        fig_box3 = px.box(df, x='Species', y='SepalWidthCm', color='Species',
                         title='Ancho del Sépalo por Especie',
                         labels={'SepalWidthCm': 'Ancho (cm)', 'Species': 'Especie'},
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_box3, use_container_width=True)
        
        fig_box4 = px.box(df, x='Species', y='PetalWidthCm', color='Species',
                         title='Ancho del Pétalo por Especie',
                         labels={'PetalWidthCm': 'Ancho (cm)', 'Species': 'Especie'},
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_box4, use_container_width=True)

# ==================== TAB 2: COMPRENSIÓN DE DATOS ====================
with tab2:
    st.header("📈 Comprensión y Análisis de los Datos")
    
    # Estadísticas descriptivas
    st.subheader("1. Estadísticas Descriptivas del Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Muestras", len(df))
    with col2:
        st.metric("Número de Características", len(df.columns) - 2)
    with col3:
        st.metric("Número de Especies", df['Species'].nunique())
    
    st.markdown("---")
    
    # Tabla de estadísticas
    st.subheader("2. Resumen Estadístico por Característica")
    stats_df = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].describe()
    st.dataframe(stats_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), 
                 use_container_width=True)
    
    st.markdown("---")
    
    # Distribución por especie
    st.subheader("3. Distribución de Muestras por Especie")
    species_count = df['Species'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(species_count, use_container_width=True)
    with col2:
        fig_pie = px.pie(values=species_count.values, names=species_count.index,
                        title='Proporción de Especies en el Dataset',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Flujo de trabajo
    st.subheader("4. Flujo de Trabajo del Proyecto")
    
    st.markdown("""
    ### 🔄 Pipeline de Data Mining Implementado:
    
    #### **Fase 1: Comprensión de los Datos**
    - 📥 Carga del dataset Iris (150 muestras, 4 características, 3 especies)
    - 🔍 Análisis exploratorio de datos (EDA)
    - 📊 Visualización de distribuciones y correlaciones
    - 📈 Identificación de patrones y características distintivas
    
    #### **Fase 2: Preparación de los Datos**
    - ✅ Verificación de valores nulos (ninguno encontrado)
    - 🔢 Codificación de variables categóricas (especies)
    - ✂️ División del dataset: 70% entrenamiento, 30% prueba
    - 📏 Las características ya están en la misma escala (cm)
    
    #### **Fase 3: Modelado**
    - 🌲 Algoritmo seleccionado: **Random Forest Classifier**
    - ⚙️ Parámetros: 100 árboles, profundidad máxima de 5
    - 🎯 Justificación: Alta precisión, maneja bien datos multiclase, proporciona feature importance
    
    #### **Fase 4: Evaluación**
    - 📊 Métricas calculadas: Accuracy, Precision, Recall, F1-Score
    - 🎯 Matriz de confusión para análisis detallado
    - 📈 Análisis de importancia de características
    
    #### **Fase 5: Despliegue**
    - 🚀 Dashboard interactivo con Streamlit
    - 🔮 Sistema de predicción en tiempo real
    - 📊 Visualización 3D y 2D de resultados
    """)
    
    st.markdown("---")
    
    # Estadísticas por especie
    st.subheader("5. Estadísticas por Especie")
    
    species_stats = df.groupby('Species')[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].mean()
    
    fig_species = go.Figure()
    for species in df['Species'].unique():
        species_data = species_stats.loc[species]
        fig_species.add_trace(go.Bar(
            name=species,
            x=['Longitud Sépalo', 'Ancho Sépalo', 'Longitud Pétalo', 'Ancho Pétalo'],
            y=species_data.values
        ))
    
    fig_species.update_layout(
        title='Comparación de Medias por Especie',
        xaxis_title='Características',
        yaxis_title='Valor Promedio (cm)',
        barmode='group'
    )
    st.plotly_chart(fig_species, use_container_width=True)

# ==================== TAB 3: ENTRENAMIENTO DEL MODELO ====================
with tab3:
    st.header("🤖 Entrenamiento y Evaluación del Modelo")
    
    # Métricas del modelo
    st.subheader("1. Métricas de Rendimiento del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>{accuracy:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎪 Precision</h3>
            <h2>{precision:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 Recall</h3>
            <h2>{recall:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚖️ F1-Score</h3>
            <h2>{f1:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("2. Importancia de las Características (Feature Importance)")
    
    feature_importance = pd.DataFrame({
        'Característica': ['Longitud del Sépalo', 'Ancho del Sépalo', 'Longitud del Pétalo', 'Ancho del Pétalo'],
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=True)
    
    fig_importance = px.bar(feature_importance, 
                           x='Importancia', 
                           y='Característica',
                           orientation='h',
                           title='Importancia de las Características en el Modelo Random Forest',
                           labels={'Importancia': 'Importancia Relativa', 'Característica': 'Características'},
                           color='Importancia',
                           color_continuous_scale='Viridis')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("""
    **💡 Interpretación:**
    - Las características de **pétalo** (longitud y ancho) son las más importantes para la clasificación
    - Esto tiene sentido biológico: los pétalos varían más entre especies que los sépalos
    - El modelo utiliza principalmente estas características para diferenciar las especies
    """)
    
    st.markdown("---")
    
    # Matriz de confusión
    st.subheader("3. Matriz de Confusión")
    
    cm = confusion_matrix(y_test, y_pred)
    species_names = le.classes_
    
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicción", y="Real", color="Cantidad"),
                       x=species_names,
                       y=species_names,
                       text_auto=True,
                       color_continuous_scale='Blues',
                       title='Matriz de Confusión del Modelo')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("---")
    
    # Reporte de clasificación
    st.subheader("4. Reporte Detallado de Clasificación")
    
    report = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.markdown("---")
    
    # Explicación del modelo
    st.subheader("5. Explicación de los Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Fortalezas del Modelo:
        - **Alta precisión** en la clasificación (>95%)
        - **Excelente separación** entre especies
        - **Bajo overfitting** gracias a la configuración del Random Forest
        - **Robusto** ante pequeñas variaciones en los datos
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Información del Entrenamiento:
        - **Algoritmo:** Random Forest
        - **Número de árboles:** 100
        - **Profundidad máxima:** 5
        - **Datos de entrenamiento:** 70% (105 muestras)
        - **Datos de prueba:** 30% (45 muestras)
        """)
    
    st.success("✨ El modelo ha sido entrenado exitosamente y está listo para realizar predicciones.")

# ==================== TAB 4: PREDICCIONES 3D ====================
with tab4:
    st.header("🎯 Sistema de Predicción con Visualización 3D")
    
    st.markdown("### Ingresa las medidas de la flor para obtener una predicción:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("🌿 Longitud del Sépalo (cm)", 
                                 float(df['SepalLengthCm'].min()), 
                                 float(df['SepalLengthCm'].max()), 
                                 5.8, 0.1)
        petal_length = st.slider("🌺 Longitud del Pétalo (cm)", 
                                 float(df['PetalLengthCm'].min()), 
                                 float(df['PetalLengthCm'].max()), 
                                 4.0, 0.1)
    
    with col2:
        sepal_width = st.slider("🍃 Ancho del Sépalo (cm)", 
                                float(df['SepalWidthCm'].min()), 
                                float(df['SepalWidthCm'].max()), 
                                3.0, 0.1)
        petal_width = st.slider("🌸 Ancho del Pétalo (cm)", 
                                float(df['PetalWidthCm'].min()), 
                                float(df['PetalWidthCm'].max()), 
                                1.2, 0.1)
    
    # Realizar predicción
    if st.button("🔮 Predecir Especie", type="primary", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        predicted_species = le.inverse_transform(prediction)[0]
        
        # Mostrar resultado
        st.markdown("---")
        st.subheader("📋 Resultado de la Predicción")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            species_emoji = {"Iris-setosa": "🌼", "Iris-versicolor": "🌺", "Iris-virginica": "🌸"}
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                <h1>{species_emoji.get(predicted_species, '🌸')}</h1>
                <h2>Especie Predicha:</h2>
                <h1 style='color: #ff4b4b;'>{predicted_species}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Probabilidades
        st.subheader("📊 Probabilidades por Especie")
        proba_df = pd.DataFrame({
            'Especie': le.classes_,
            'Probabilidad': prediction_proba[0]
        }).sort_values('Probabilidad', ascending=False)
        
        fig_proba = px.bar(proba_df, x='Especie', y='Probabilidad',
                          title='Confianza de la Predicción',
                          color='Probabilidad',
                          color_continuous_scale='Reds',
                          text=proba_df['Probabilidad'].apply(lambda x: f'{x:.2%}'))
        fig_proba.update_traces(textposition='outside')
        st.plotly_chart(fig_proba, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 3D
        st.subheader("🎨 Visualización 3D: Posición de la Muestra")
        
        df_plot = df.copy()
        df_plot['Tipo'] = 'Dataset'
        
        new_point = pd.DataFrame({
            'SepalLengthCm': [sepal_length],
            'SepalWidthCm': [sepal_width],
            'PetalLengthCm': [petal_length],
            'PetalWidthCm': [petal_width],
            'Species': [predicted_species],
            'Tipo': ['Predicción']
        })
        
        df_combined = pd.concat([df_plot, new_point], ignore_index=True)
        
        fig_3d = px.scatter_3d(df_combined, 
                              x='SepalLengthCm', 
                              y='PetalLengthCm', 
                              z='PetalWidthCm',
                              color='Species',
                              symbol='Tipo',
                              title='Distribución 3D de Especies Iris con Punto Predicho',
                              labels={
                                  'SepalLengthCm': 'Longitud Sépalo (cm)',
                                  'PetalLengthCm': 'Longitud Pétalo (cm)',
                                  'PetalWidthCm': 'Ancho Pétalo (cm)'
                              },
                              color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                              size_max=15)
        
        fig_3d.update_traces(marker=dict(size=5), selector=dict(name='Dataset'))
        fig_3d.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), 
                            selector=dict(name='Predicción'))
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💡 **Tip:** Puedes rotar el gráfico 3D arrastrando con el mouse para ver diferentes ángulos.")

# ==================== TAB 5: PREDICCIONES 2D ====================
with tab5:
    st.header("📉 Visualización 2D de Predicciones")
    
    st.markdown("### Ingresa las medidas de la flor:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length_2d = st.slider("🌿 Longitud del Sépalo (cm) ", 
                                    float(df['SepalLengthCm'].min()), 
                                    float(df['SepalLengthCm'].max()), 
                                    5.8, 0.1, key='sl2d')
        petal_length_2d = st.slider("🌺 Longitud del Pétalo (cm) ", 
                                    float(df['PetalLengthCm'].min()), 
                                    float(df['PetalLengthCm'].max()), 
                                    4.0, 0.1, key='pl2d')
    
    with col2:
        sepal_width_2d = st.slider("🍃 Ancho del Sépalo (cm) ", 
                                   float(df['SepalWidthCm'].min()), 
                                   float(df['SepalWidthCm'].max()), 
                                   3.0, 0.1, key='sw2d')
        petal_width_2d = st.slider("🌸 Ancho del Pétalo (cm) ", 
                                   float(df['PetalWidthCm'].min()), 
                                   float(df['PetalWidthCm'].max()), 
                                   1.2, 0.1, key='pw2d')
    
    # Realizar predicción
    if st.button("🔮 Predecir Especie ", type="primary", use_container_width=True, key='predict2d'):
        input_data = np.array([[sepal_length_2d, sepal_width_2d, petal_length_2d, petal_width_2d]])
        prediction = model.predict(input_data)
        predicted_species = le.inverse_transform(prediction)[0]
        
        st.markdown("---")
        st.success(f"✅ **Especie Predicha:** {predicted_species}")
        st.markdown("---")
        
        # Crear múltiples gráficos 2D
        st.subheader("📊 Gráficos 2D de Dispersión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico 1: Pétalo (Longitud vs Ancho)
            df_plot = df.copy()
            df_plot['Tipo'] = 'Dataset'
            
            new_point = pd.DataFrame({
                'PetalLengthCm': [petal_length_2d],
                'PetalWidthCm': [petal_width_2d],
                'Species': [predicted_species],
                'Tipo': ['Predicción']
            })
            
            df_combined = pd.concat([df_plot[['PetalLengthCm', 'PetalWidthCm', 'Species', 'Tipo']], new_point], ignore_index=True)
            
            fig_2d_1 = px.scatter(df_combined,
                                 x='PetalLengthCm',
                                 y='PetalWidthCm',
                                 color='Species',
                                 symbol='Tipo',
                                 title='Características del Pétalo',
                                 labels={
                                     'PetalLengthCm': 'Longitud del Pétalo (cm)',
                                     'PetalWidthCm': 'Ancho del Pétalo (cm)'
                                 },
                                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            fig_2d_1.update_traces(marker=dict(size=8), selector=dict(mode='markers', name='Dataset'))
            fig_2d_1.update_traces(marker=dict(size=20, line=dict(width=3, color='black')), 
                                  selector=dict(name='Predicción'))
            
            st.plotly_chart(fig_2d_1, use_container_width=True)
            
            # Gráfico 3: Sépalo vs Pétalo (Longitud)
            new_point_3 = pd.DataFrame({
                'SepalLengthCm': [sepal_length_2d],
                'PetalLengthCm': [petal_length_2d],
                'Species': [predicted_species],
                'Tipo': ['Predicción']
            })
            
            df_combined_3 = pd.concat([df_plot[['SepalLengthCm', 'PetalLengthCm', 'Species', 'Tipo']], new_point_3], ignore_index=True)
            
            fig_2d_3 = px.scatter(df_combined_3,
                                 x='SepalLengthCm',
                                 y='PetalLengthCm',
                                 color='Species',
                                 symbol='Tipo',
                                 title='Longitud: Sépalo vs Pétalo',
                                 labels={
                                     'SepalLengthCm': 'Longitud del Sépalo (cm)',
                                     'PetalLengthCm': 'Longitud del Pétalo (cm)'
                                 },
                                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            fig_2d_3.update_traces(marker=dict(size=8), selector=dict(mode='markers', name='Dataset'))
            fig_2d_3.update_traces(marker=dict(size=20, line=dict(width=3, color='black')), 
                                  selector=dict(name='Predicción'))
            
            st.plotly_chart(fig_2d_3, use_container_width=True)
        
        with col2:
            # Gráfico 2: Sépalo (Longitud vs Ancho)
            new_point_2 = pd.DataFrame({
                'SepalLengthCm': [sepal_length_2d],
                'SepalWidthCm': [sepal_width_2d],
                'Species': [predicted_species],
                'Tipo': ['Predicción']
            })
            
            df_combined_2 = pd.concat([df_plot[['SepalLengthCm', 'SepalWidthCm', 'Species', 'Tipo']], new_point_2], ignore_index=True)
            
            fig_2d_2 = px.scatter(df_combined_2,
                                 x='SepalLengthCm',
                                 y='SepalWidthCm',
                                 color='Species',
                                 symbol='Tipo',
                                 title='Características del Sépalo',
                                 labels={
                                     'SepalLengthCm': 'Longitud del Sépalo (cm)',
                                     'SepalWidthCm': 'Ancho del Sépalo (cm)'
                                 },
                                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            fig_2d_2.update_traces(marker=dict(size=8), selector=dict(mode='markers', name='Dataset'))
            fig_2d_2.update_traces(marker=dict(size=20, line=dict(width=3, color='black')), 
                                  selector=dict(name='Predicción'))
            
            st.plotly_chart(fig_2d_2, use_container_width=True)
            
            # Gráfico 4: Ancho Sépalo vs Ancho Pétalo
            new_point_4 = pd.DataFrame({
                'SepalWidthCm': [sepal_width_2d],
                'PetalWidthCm': [petal_width_2d],
                'Species': [predicted_species],
                'Tipo': ['Predicción']
            })
            
            df_combined_4 = pd.concat([df_plot[['SepalWidthCm', 'PetalWidthCm', 'Species', 'Tipo']], new_point_4], ignore_index=True)
            
            fig_2d_4 = px.scatter(df_combined_4,
                                 x='SepalWidthCm',
                                 y='PetalWidthCm',
                                 color='Species',
                                 symbol='Tipo',
                                 title='Ancho: Sépalo vs Pétalo',
                                 labels={
                                     'SepalWidthCm': 'Ancho del Sépalo (cm)',
                                     'PetalWidthCm': 'Ancho del Pétalo (cm)'
                                 },
                                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            fig_2d_4.update_traces(marker=dict(size=8), selector=dict(mode='markers', name='Dataset'))
            fig_2d_4.update_traces(marker=dict(size=20, line=dict(width=3, color='black')), 
                                  selector=dict(name='Predicción'))
            
            st.plotly_chart(fig_2d_4, use_container_width=True)
        
        st.info("💡 **Interpretación:** El punto grande con borde negro representa tu predicción. Observa cómo se posiciona respecto a las diferentes especies.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #666;'>
        🌸 <b>Proyecto Final de Data Mining</b> 🌸<br>
        Universidad de la Costa - 2024<br>
        Desarrollado con ❤️ usando Streamlit y Python
    </p>
</div>
""", unsafe_allow_html=True)