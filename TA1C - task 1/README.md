# 📰 TA1C @ IberLEF 2025 — Detección de Clickbait de Twitters en Español

> 🧠 *"Te Ahorré Un Click": un desafío compartido sobre detección y resolución de clickbait en redes sociales*  
> Participación en la Tarea 1 — Clasificación Binaria de Clickbait

---

## 🧩 Descripción del desafío

El fenómeno del **clickbait** afecta significativamente la calidad de la información en medios digitales. Titulares diseñados para generar curiosidad desproporcionada inducen a los usuarios a hacer clic, aunque el contenido real no siempre cumple con lo prometido. 

La tarea TA1C, parte de **IberLEF 2025**, propone abordar este problema desde una perspectiva automatizada, evaluando sistemas capaces de:

- 🧪 **Tarea 1: Detección de Clickbait**  
  Clasificar si un tweet que enlaza a una noticia en español es clickbait o no, utilizando la teoría del "gap de información" de Loewenstein (1994) como base teórica.  
  La evaluación se realiza mediante **Accuracy, Precision, Recall** y **F1-score** (principal métrica).

---

## 💼 Sobre este repositorio

Este repositorio contiene mi participación en la **Tarea 1 (Clickbait Detection)** del reto TA1C. En él se documenta el desarrollo, implementación y evaluación de modelos de clasificación binaria para detectar clickbait en tweets escritos en español.

### 🔍 Introducción del trabajo

La detección de clickbait ha ganado relevancia en el campo de la Inteligencia Artificial, debido al impacto que tiene sobre la veracidad de la información en línea. En este estudio, se desarrollaron tres modelos con diferentes enfoques:

- 📘 RoBERTuito + features manuales  
- ⚖️ RoBERTuito con ponderación de clases  
- 🦙 Llama 3.2 – 3B

Se utilizaron los conjuntos de datos proporcionados por los organizadores del reto, y se abordaron desafíos clave como el **desbalance de clases** y la **variabilidad del lenguaje en redes sociales**. El objetivo fue identificar el modelo más eficaz mediante el análisis de métricas de desempeño (precisión, recall, F1).

---

## 📈 Resultados destacados

El modelo basado en **Llama 3.2 – 3B** obtuvo el mejor rendimiento global, con un **F1-score del 95.04%**, superando a los otros dos modelos. Esto resalta la capacidad de los modelos de lenguaje avanzados para adaptarse a tareas complejas en español.

---

## 📚 Créditos y referencias

Este trabajo se enmarca en la competencia [IberLEF 2025](https://sites.google.com/view/iberlef-2025/tasks?authuser=0) y está basado en la definición de clickbait propuesta por Mordecki et al. (2024).  
Agradecimientos a los organizadores del reto TA1C por proveer los datasets y lineamientos.

---

## 🚀 Tecnologías utilizadas

- Python 3.10  
- Transformers (HuggingFace)  
- Scikit-learn  
- PyTorch  
- Google Colab / Jupyter Notebooks

---


