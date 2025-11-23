# app/services/ml_service.py

import joblib
import pandas as pd
from typing import Dict, List, Any
from app.core.config import settings

class MLService:
    model = None

    @classmethod
    def load_model(cls):
        """Carrega o modelo do disco para a memória."""
        if cls.model is None:
            try:
                cls.model = joblib.load(settings.MODEL_PATH)
                print("Modelo de ML carregado com sucesso.")
            except FileNotFoundError:
                print(f"Erro: Arquivo do modelo não encontrado em {settings.MODEL_PATH}")
                cls.model = None
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
                cls.model = None
    
    @classmethod
    def is_model_loaded(cls) -> bool:
        """Verifica se o modelo foi carregado."""
        return cls.model is not None

    @classmethod
    def predict_proba(cls, features: pd.DataFrame) -> List[float]:
        """
        Realiza a predição da probabilidade de evasão.
        
        Args:
            features (pd.DataFrame): DataFrame com as features do(s) aluno(s).
        
        Returns:
            List[float]: Lista com as probabilidades de evasão (classe 1).
        """
        if not cls.is_model_loaded():
            raise RuntimeError("Modelo de ML não foi carregado. A predição não pode ser realizada.")
        
        # A predição de probabilidade retorna um array com [prob_classe_0, prob_classe_1]
        # Queremos apenas a probabilidade da classe 1 (evasão)
        probabilities = cls.model.predict_proba(features)[:, 1]
        return probabilities.tolist()

# Instanciando o serviço para que o modelo possa ser carregado na inicialização
ml_service = MLService()