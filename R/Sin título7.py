#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################
# Clasificador Naive Bayes en Python #
####################################

# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Vamos a utilizar un dataset con 435 instancias
# de las cuales 61.37931% son votantes demócratas y 38.62069% son republicanos

###########################
# 1. Preparación de datos #
###########################

# Definir nombres de columnas
header = ["NAME","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16"]

# Leer los datos sin header
vote_data = pd.read_csv("house-votes-84.data", header=None, names=header)

# Observar el formato del dataset
print("Primeras filas:")
print(vote_data.head())
print("\nÚltimas filas:")
print(vote_data.tail())

# Reemplazar '?' por valores vacíos (NaN)
vote_data = vote_data.replace('?', np.nan)

print("\nEstructura del dataset:")
print(vote_data.info())

# Verificar las proporciones de cada clase
print("\nProporciones de las clases:")
print(vote_data['NAME'].value_counts(normalize=True))

##############################################
# 2. Creación de datos de entrenamiento/test #
##############################################

vote_raw_train = vote_data.iloc[0:370, :]
vote_raw_test = vote_data.iloc[370:435, :]

# Observar que se mantienen las proporciones
print("\nProporciones en entrenamiento:")
print(vote_raw_train['NAME'].value_counts(normalize=True))
print("\nProporciones en test:")
print(vote_raw_test['NAME'].value_counts(normalize=True))

##########################################
# 3. Creación de features para el modelo #
##########################################

# Preparar los datos para el clasificador
# Convertir las variables categóricas a numéricas
# y=1, n=0, NaN se mantiene como NaN

def encode_data(df):
    df_encoded = df.copy()
    # Codificar todas las columnas excepto NAME
    for col in df_encoded.columns[1:]:
        df_encoded[col] = df_encoded[col].map({'y': 1, 'n': 0})
    return df_encoded

vote_train_encoded = encode_data(vote_raw_train)
vote_test_encoded = encode_data(vote_raw_test)

# Separar características y etiquetas
X_train = vote_train_encoded.drop('NAME', axis=1)
y_train = vote_train_encoded['NAME']
X_test = vote_test_encoded.drop('NAME', axis=1)
y_test = vote_test_encoded['NAME']

# Manejar valores faltantes: rellenar con la moda de cada columna
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Entrenar un clasificador Naive Bayes
# Usamos CategoricalNB que maneja variables categóricas
# alpha=1 equivale a laplace=1 en R (suavizado de Laplace)
from sklearn.naive_bayes import BernoulliNB
vote_classifier = BernoulliNB(alpha=1.0)
vote_classifier.fit(X_train_imputed, y_train)

# Predecir la clase más probable
vote_test_pred = vote_classifier.predict(X_test_imputed)

print("\n" + "="*50)
print("RESULTADOS DEL MODELO")
print("="*50)

# Matriz de confusión
print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, vote_test_pred)
cm_df = pd.DataFrame(cm, 
                     index=['democrat (real)', 'republican (real)'], 
                     columns=['democrat (pred)', 'republican (pred)'])
print(cm_df)

# Proporciones en la matriz de confusión
print("\nMatriz de confusión (proporciones):")
cm_prop = confusion_matrix(y_test, vote_test_pred, normalize='true')
cm_prop_df = pd.DataFrame(cm_prop, 
                          index=['democrat (real)', 'republican (real)'], 
                          columns=['democrat (pred)', 'republican (pred)'])
print(cm_prop_df)

# Predicciones donde se ha equivocado el modelo
print("\nPredicciones incorrectas:")
incorrect = vote_raw_test[y_test != vote_test_pred]
print(f"Total de errores: {len(incorrect)}")
print(incorrect)

# Predecir con probabilidades
vote_test_prob = vote_classifier.predict_proba(X_test_imputed)
pred_df = pd.DataFrame(vote_test_prob, 
                       columns=vote_classifier.classes_)
print("\nPrimeras probabilidades predichas:")
print(pred_df.head())

####################################
# 4. Curvas de evaluación del modelo #
####################################

# Obtener las probabilidades para la clase 'democrat'
y_test_binary = (y_test == 'democrat').astype(int)
prob_democrat = pred_df['democrat']

# Calcular curvas
fpr, tpr, thresholds_roc = roc_curve(y_test_binary, prob_democrat)
precision, recall, thresholds_pr = precision_recall_curve(y_test_binary, prob_democrat)

# Calcular sensitivity y specificity
sensitivity = tpr
specificity = 1 - fpr

# Crear figura con las tres gráficas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Curva ROC
axes[0].plot(fpr, tpr, linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
axes[0].set_title('Curva ROC', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Curva Precision-Recall
axes[1].plot(recall, precision, linewidth=2)
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Curva Sensitivity-Specificity
axes[2].plot(specificity, sensitivity, linewidth=2)
axes[2].set_xlabel('Specificity', fontsize=12)
axes[2].set_ylabel('Sensitivity', fontsize=12)
axes[2].set_title('Curva Sensitivity-Specificity', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([1, 0])  # Invertir eje x para que coincida con el formato tradicional

plt.tight_layout()
plt.show()

# Calcular métricas adicionales
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print("\n" + "="*50)
print("MÉTRICAS DE EVALUACIÓN")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, vote_test_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test_binary, prob_democrat):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, vote_test_pred))


# In[ ]:




