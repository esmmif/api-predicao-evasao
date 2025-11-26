# TREINAMENTO E AVALIAÇÃO DO MODELO

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os

def train_and_evaluate():
    """
    Função principal para ler os dados, pré-processar, treinar,
    avaliar e salvar o modelo de regressão logística.
    """
    # 1. Leitura dos dados
    print("Lendo os dados...")
    try:
        df = pd.read_csv('data/alunos.csv')
    except FileNotFoundError:
        print("Arquivo 'alunos.csv' não encontrado. Gere o dataset primeiro.")
        return

    # Definindo features e target
    X = df.drop('evasao_ate_1ano', axis=1)
    y = df['evasao_ate_1ano']
    
    # Identificar tipos de colunas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # 2. Limpeza / Preparação dos dados (usando Pipelines)
    print("Construindo pipeline de pré-processamento...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Tratar ausentes
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Codificar categóricas
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Divisão em treino e teste
    print("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y # Estratificado
    )

    # 4. Treinamento do modelo de Regressão Logística
    print("Treinando o modelo de Regressão Logística...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)

    # 5. Avaliação do modelo
    print("\nAvaliando o modelo...")
    y_pred = model_pipeline.predict(X_test)
    
    print("\nRelatório de Classificação:")
    # Acurácia, Precision, Recall, F1-score
    print(classification_report(y_test, y_pred))

    # AUC-ROC
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Plotar curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    # 6. Salvando o modelo treinado
    print("\nSalvando o modelo em disco...")
    output_dir = 'model'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'logistic_model.pkl')
    joblib.dump(model_pipeline, model_path)
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":

    if not os.path.exists('data/alunos.csv'):
        os.makedirs('data', exist_ok=True)

        print("Por favor, crie o arquivo data/alunos.csv ou use a função para gerá-lo.")
    else:
        train_and_evaluate()
