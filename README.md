
# 🧪 Detección de Fake News y Clasificación de Imágenes de Moda

Este repositorio contiene dos notebooks desarrollados como parte de a carrera de Ciencia de Datos de Coderhouse. Cada uno aborda una problemática diferente utilizando técnicas de aprendizaje automático y profundo.

---

## 📘 Proyecto 1: Detección de Fake News (`01_fake_news.ipynb`)

Este notebook implementa un clasificador de noticias falsas mediante técnicas de Procesamiento de Lenguaje Natural (NLP) y aprendizaje supervisado.

### 🔍 Descripción
- **Objetivo:** Clasificar noticias como reales o falsas.
- **Dataset:** `fake_or_real_news.csv` (obtenido desde GitHub).
- **Técnicas utilizadas:**
  - Limpieza y preprocesamiento de texto con `nltk` y `spaCy`.
  - Visualización de datos con `seaborn`, `matplotlib`, y `WordCloud`.
  - Modelado con un pipeline `TfidfVectorizer` + `MultinomialNB` de `sklearn`.

### 🛠️ Requisitos
- `pandas`, `numpy`, `nltk`, `spacy`, `matplotlib`, `seaborn`, `scikit-learn`, `wordcloud`

### 🧠 Modelo
- Clasificador: Naive Bayes
- Métricas: Accuracy, Reporte de Clasificación

---

## 🧠 Proyecto 2: Clasificación de Imágenes de Moda (`02_proyecto_final.ipynb`)

Este notebook entrena una red neuronal profunda para reconocer artículos de ropa a partir de imágenes del dataset **Fashion MNIST**.

### 🔍 Descripción
- **Objetivo:** Clasificar imágenes de ropa en 10 categorías (camisetas, zapatos, abrigos, etc.).
- **Dataset:** `fashion_mnist` (disponible desde `tensorflow.keras.datasets`).
- **Técnicas utilizadas:**
  - Construcción de una red neuronal secuencial.
  - Capas densas, `Dropout`, `Flatten`.
  - Entrenamiento y evaluación con validación cruzada.

### 🛠️ Herramientas utilizadas
- `tensorflow`, `matplotlib`, `tensorflow_datasets`
