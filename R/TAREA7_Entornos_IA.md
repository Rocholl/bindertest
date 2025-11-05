# TAREA 7 - Entornos de Programación de IA con Python y R

**Autor:** [Tu Nombre]  
**Fecha:** [Fecha]  
**Asignatura:** [Nombre de la Asignatura]

---

## Índice

1. [Introducción](#introducción)
2. [Dataset y Modelo](#dataset-y-modelo)
3. [Entornos de Desarrollo](#entornos-de-desarrollo)
4. [Comparativa de Resultados](#comparativa-de-resultados)
5. [Conclusiones](#conclusiones)

---

## Introducción

Esta tarea evalúa el modelo **Naive Bayes** en múltiples entornos de desarrollo utilizando **Python** y **R**, calculando métricas de evaluación como el **AUC-ROC** (Área bajo la curva ROC).

### Objetivo

Comparar el rendimiento del modelo Naive Bayes en diferentes entornos de programación, evaluando métricas como Accuracy, Precision, Recall, F1-Score y AUC-ROC.

---

## Dataset y Modelo

### Dataset
- **Nombre:** [Nombre del dataset]
- **Fuente:** [URL o referencia]
- **Instancias:** [número]
- **Atributos:** [número]
- **Variable objetivo:** [nombre]
- **Tipo:** Clasificación

### Modelo
- **Algoritmo:** Naive Bayes
- **Métricas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
---

## Entornos de Desarrollo

### 1. Visual Studio Code
- **Lenguaje:** Python / R
- **Descripción:** Editor ligero y potente de Microsoft
- **Captura:** ![VSCode](capturas/vscode_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 2. RStudio
- **Lenguaje:** R
- **Descripción:** IDE especializado en análisis estadístico
- **Captura:** ![RStudio](capturas/rstudio_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 3. Google Colab
- **Lenguaje:** Python
- **Descripción:** Jupyter Notebook en la nube con GPUs gratuitas
- **Captura:** ![Colab](capturas/colab_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 4. Jupyter Notebook (Anaconda)
- **Lenguaje:** Python
- **Descripción:** Notebooks interactivos locales
- **Captura:** ![Jupyter](capturas/jupyter_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 5. Spyder (Anaconda)
- **Lenguaje:** Python
- **Descripción:** IDE científico para Python
- **Captura:** ![Spyder](capturas/spyder_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 6. PyCharm
- **Lenguaje:** Python
- **Descripción:** IDE profesional de JetBrains
- **Captura:** ![PyCharm](capturas/pycharm_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____

---

### 7. Kaggle
- **Lenguaje:** Python
- **Descripción:** Plataforma en línea con GPUs y datasets públicos
- **Captura:** ![Kaggle](capturas/kaggle_entorno.png)

**Resultados:**
- Accuracy: _____
- Precision: _____
- Recall: _____
- F1-Score: _____
- **AUC-ROC:** _____
---

## Comparativa de Resultados

### Tabla Comparativa

| Entorno | Lenguaje | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Tiempo (s) |
|---------|----------|----------|-----------|--------|----------|---------|------------|
| VS Code | Python/R | _____ | _____ | _____ | _____ | _____ | _____ |
| RStudio | R | _____ | _____ | _____ | _____ | _____ | _____ |
| Google Colab | Python | _____ | _____ | _____ | _____ | _____ | _____ |
| Jupyter | Python | _____ | _____ | _____ | _____ | _____ | _____ |
| Spyder | Python | _____ | _____ | _____ | _____ | _____ | _____ |
| PyCharm | Python | _____ | _____ | _____ | _____ | _____ | _____ |
| Kaggle | Python | _____ | _____ | _____ | _____ | _____ | _____ |

### Análisis
- **Media AUC-ROC:** _____
- **Desviación estándar:** _____
- **Mejor entorno (AUC):** _____
- **Mejor entorno (tiempo):** _____

---

## Conclusiones

### Ventajas y Desventajas

**Python:**
- ✅ Ventajas: [Completar]
- ❌ Desventajas: [Completar]

**R:**
- ✅ Ventajas: [Completar]
- ❌ Desventajas: [Completar]

### Conclusión General
[Escribe tus conclusiones aquí]

### Recomendaciones
[Escribe tus recomendaciones aquí]
---

## Anexos

### Código Python (Ejemplo)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Cargar dataset
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métricas
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1-Score: {f1_score(y_test, y_pred):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}')
```

### Código R (Ejemplo)

```r
library(e1071)
library(caret)
library(pROC)

# Cargar dataset
data <- read.csv('dataset.csv')

# Dividir datos
set.seed(42)
trainIndex <- createDataPartition(data$target, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Entrenar modelo
model <- naiveBayes(target ~ ., data = train_data)

# Predicciones
predictions <- predict(model, test_data)
probabilities <- predict(model, test_data, type = "raw")

# Métricas
confusionMatrix(predictions, test_data$target)
roc_obj <- roc(test_data$target, probabilities[, 2])
print(paste("AUC-ROC:", auc(roc_obj)))
```

---

## Referencias

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [e1071 R Package](https://cran.r-project.org/package=e1071)
3. [Documentación sobre métricas de clasificación]

---

**Fin del documento**

