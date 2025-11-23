# 🌸 Proyecto de Clasificación de Especies Iris

## Descripción del Proyecto

Este proyecto implementa un sistema completo de clasificación de especies de flores Iris utilizando técnicas de Data Mining y Machine Learning. El dashboard interactivo permite visualizar datos, comprender el proceso de entrenamiento y realizar predicciones en tiempo real.

## 👥 Equipo de Desarrollo

- **Estudiante 1:** [ALEJANDRO ESCORCIA]
- **Estudiante 2:** [ASHLEY URUETA]

**Profesor:** José Escorcia-Gutierrez, Ph.D.  
**Institución:** Universidad de la Costa  
**Curso:** Data Mining  
**Año:** 2024

## 📋 Características del Proyecto

### Dataset
El proyecto utiliza el famoso dataset Iris que contiene:
- **150 muestras** de flores
- **3 especies:** Iris-setosa, Iris-versicolor, Iris-virginica
- **4 características:** Longitud del sépalo, ancho del sépalo, longitud del pétalo, ancho del pétalo

### Funcionalidades del Dashboard

1. **📊 Visualización de Datos**
   - Histogramas de distribución por característica
   - Mapa de calor de correlaciones
   - Boxplots por especie
   - Análisis exploratorio completo

2. **📈 Comprensión de los Datos**
   - Estadísticas descriptivas detalladas
   - Distribución de muestras por especie
   - Explicación del flujo de trabajo
   - Análisis estadístico por especie

3. **🤖 Entrenamiento del Modelo**
   - Métricas de rendimiento (Accuracy, Precision, Recall, F1-Score)
   - Gráfica de Feature Importance
   - Matriz de confusión
   - Reporte detallado de clasificación

4. **🎯 Predicciones con Visualización 3D**
   - Entrada interactiva de características
   - Predicción en tiempo real
   - Gráfico 3D de dispersión
   - Probabilidades de clasificación

5. **📉 Predicciones con Visualización 2D**
   - Múltiples vistas 2D de las características
   - Comparación visual con el dataset
   - Análisis detallado de la posición de la muestra


### Pipeline de Data Mining Implementado

1. **Comprensión de los Datos**
   - Carga y exploración del dataset
   - Análisis de distribuciones y correlaciones
   - Identificación de patrones

2. **Preparación de los Datos**
   - Verificación de valores nulos
   - Codificación de variables categóricas
   - División en conjunto de entrenamiento (70%) y prueba (30%)

3. **Modelado**
   - Algoritmo: Random Forest Classifier
   - Parámetros: 100 árboles, profundidad máxima de 5
   - Justificación: Alta precisión, manejo de datos multiclase

4. **Evaluación**
   - Métricas: Accuracy, Precision, Recall, F1-Score
   - Matriz de confusión
   - Análisis de Feature Importance

5. **Despliegue**
   - Dashboard interactivo con Streamlit
   - Sistema de predicción en tiempo real
   - Visualizaciones 3D y 2D

## 🎯 Resultados

El modelo Random Forest alcanza una precisión superior al **95%** en la clasificación de especies de Iris, demostrando:
- Excelente separación entre especies
- Alta confiabilidad en las predicciones
- Robustez ante variaciones en los datos

### Importancia de Características
1. **Longitud del Pétalo** - Mayor importancia
2. **Ancho del Pétalo** - Alta importancia
3. **Longitud del Sépalo** - Importancia media
4. **Ancho del Sépalo** - Menor importancia

## 💻 Tecnologías Utilizadas

- **Python 3.x** - Lenguaje de programación
- **Streamlit** - Framework para el dashboard interactivo
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Scikit-learn** - Algoritmos de Machine Learning
- **Plotly** - Visualizaciones interactivas
- **Matplotlib & Seaborn** - Visualizaciones estadísticas


## 🎥 Video de Presentación

[Enlace al video de presentación del proyecto]

