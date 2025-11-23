# 🌸 Dashboard de Clasificación de Especies de Iris

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de machine learning para clasificar especies de flores de iris utilizando el clásico dataset de Iris. La aplicación cuenta con un dashboard interactivo construido con Streamlit que permite a los usuarios explorar los datos, comprender el rendimiento del modelo y hacer predicciones.

**Curso**: Minería de Datos  
**Institución**: Universidad de la Costa  
**Instructor**: José Escorcia-Gutierrez, Ph.D.

## Miembros del Equipo

- [Nombre Miembro 1]
- [Nombre Miembro 2]
- [Nombre Miembro 3]
- [Nombre Miembro 4]

## Descripción del Dataset

El dataset de Iris contiene 150 muestras de flores de iris de tres especies:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

Cada muestra incluye cuatro características:
- Longitud del Sépalo (cm)
- Ancho del Sépalo (cm)
- Longitud del Pétalo (cm)
- Ancho del Pétalo (cm)

## Metodología

### 1. Comprensión de Datos
- Análisis exploratorio de datos (EDA)
- Resumen estadístico
- Análisis de distribución por especie
- Análisis de correlación entre características

### 2. Preprocesamiento de Datos
- Escalado de características usando `StandardScaler`
- División entrenamiento-prueba (80-20) con estratificación para mantener balance de clases

### 3. Selección del Modelo
**Algoritmo**: Random Forest Classifier (Clasificador de Bosque Aleatorio)

**Justificación**: Random Forest es ideal para este problema porque:
- Maneja relaciones no lineales entre características
- Es robusto ante valores atípicos
- Proporciona rankings de importancia de características
- Requiere mínimo ajuste de hiperparámetros
- Excelente rendimiento en datasets pequeños a medianos
- Bajo riesgo de sobreajuste con configuración adecuada
- No necesita que las características sigan una distribución específica
- Puede capturar interacciones complejas entre variables

**Configuración del Modelo**:
- Número de estimadores: 100 árboles
- Profundidad máxima: 5 niveles
- Estado aleatorio: 42 (para reproducibilidad)

### 4. Evaluación del Modelo
El modelo se evalúa utilizando múltiples métricas:
- **Exactitud (Accuracy)**: Corrección general de las predicciones
- **Precisión (Precision)**: Proporción de predicciones positivas correctas
- **Recall (Sensibilidad)**: Proporción de positivos reales identificados correctamente
- **F1-Score**: Media armónica de precisión y recall
- **Matriz de Confusión**: Desglose detallado de predicciones vs valores reales

### 5. Despliegue
Dashboard interactivo de Streamlit con cuatro secciones principales:
- Inicio: Descripción del proyecto y estadísticas rápidas
- Exploración de Datos: Visualizaciones y análisis estadístico
- Rendimiento del Modelo: Métricas y evaluación
- Hacer Predicciones: Interfaz de predicción interactiva con visualización 3D

## Características del Dashboard

### Páginas del Dashboard

#### 🏠 Inicio
- Descripción general del proyecto
- Descripción del dataset
- Flujo de trabajo de la metodología
- Estadísticas rápidas
- Justificación del modelo seleccionado

#### 📈 Exploración de Datos
- Resumen del dataset y estadísticas descriptivas
- Visualizaciones de distribución de clases
- Distribuciones de características por especie
- Mapa de calor de correlaciones
- Matriz de dispersión para relaciones entre pares

#### 🤖 Rendimiento del Modelo
- Métricas de rendimiento (Exactitud, Precisión, Recall, F1-Score)
- Matriz de confusión
- Visualización de importancia de características
- Detalles de configuración del modelo
- Interpretación de métricas

#### 🔮 Hacer Predicciones
- Controles deslizantes interactivos para medidas de flores
- Predicción de especies en tiempo real
- Niveles de confianza para cada clase
- Gráfico de dispersión 3D mostrando la predicción en contexto del dataset
- Interpretación visual de resultados

## Instalación

### Prerrequisitos
- Python 3.8 o superior
- Gestor de paquetes pip

### Instrucciones de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/[tu-usuario]/iris-classification.git
cd iris-classification
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar los paquetes requeridos:
```bash
pip install -r requirements.txt
```

## Uso

Ejecutar la aplicación Streamlit:
```bash
streamlit run Proyect.py
```

El dashboard se abrirá automáticamente en tu navegador predeterminado en `http://localhost:8501`

## Estructura del Proyecto

```
iris-classification/
│
├── Proyect.py              # Aplicación principal de Streamlit
├── requirements.txt        # Dependencias de Python
├── README.md              # Documentación del proyecto
└── [otros archivos]       # Recursos adicionales
```

## Rendimiento del Modelo

El modelo Random Forest logra un excelente rendimiento en el dataset de Iris:
- **Alta exactitud** (típicamente >95%)
- **Rendimiento balanceado** en las tres especies
- **Fuerte importancia de características** de las medidas de pétalos

## Stack Tecnológico

- **Python**: Lenguaje de programación principal
- **Streamlit**: Framework para dashboard web
- **scikit-learn**: Biblioteca de machine learning
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Plotly**: Visualizaciones interactivas
- **Matplotlib & Seaborn**: Visualizaciones estadísticas

## Presentación en Video

[El enlace a la presentación en video se agregará aquí]

La presentación en video cubre:
1. Metodología y diseño del flujo de trabajo
2. Justificación de las decisiones técnicas
3. Demostración del dashboard
4. Explicación de la visualización de predicciones

## Flujo de Trabajo del Pipeline

```
1. Carga de Datos
   ↓
2. Análisis Exploratorio
   ↓
3. Preprocesamiento (Escalado)
   ↓
4. División Train-Test (80-20)
   ↓
5. Entrenamiento del Modelo
   ↓
6. Evaluación con Métricas
   ↓
7. Visualización de Resultados
   ↓
8. Predicción Interactiva
```

## Decisiones de Diseño

### ¿Por qué Random Forest?
1. **Robustez**: Maneja bien ruido y valores atípicos
2. **Interpretabilidad**: Proporciona importancia de características
3. **Precisión**: Alto rendimiento sin ajuste extenso
4. **Versatilidad**: No requiere preprocesamiento complejo
5. **Estabilidad**: Múltiples árboles reducen varianza

### ¿Por qué StandardScaler?
1. **Mejora convergencia**: Ayuda a algoritmos basados en distancia
2. **Equidad de características**: Todas las características tienen igual peso inicial
3. **Rendimiento**: Mejora la velocidad de entrenamiento
4. **Estándar de industria**: Práctica común en ML

## Referencias

- Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems"
- UCI Machine Learning Repository: Iris Dataset
- Documentación de scikit-learn
- Documentación de Streamlit
- Breiman, L. (2001). "Random Forests". Machine Learning

## Licencia

Este proyecto se crea con fines educativos como parte del curso de Minería de Datos en la Universidad de la Costa.

## Agradecimientos

Agradecimiento especial al Profesor José Escorcia-Gutierrez, Ph.D. por la guía durante el curso y el desarrollo del proyecto.

---

## Guía para la Presentación en Video

### Estructura Sugerida (5-7 minutos):

1. **Introducción (30 seg)**
   - Presentación del equipo
   - Objetivo del proyecto

2. **Metodología (2 min)**
   - Explicar el flujo de trabajo paso a paso
   - Justificar la elección de Random Forest
   - Explicar el preprocesamiento

3. **Demostración del Dashboard (3 min)**
   - Mostrar página de inicio
   - Exploración de datos (distribuciones, correlaciones)
   - Métricas del modelo
   - Hacer una predicción en vivo

4. **Conclusiones (30 seg)**
   - Resultados obtenidos
   - Aprendizajes del proyecto

---

*"Las tres principales virtudes de un programador son: Pereza, Impaciencia y Arrogancia." - Larry Wall*