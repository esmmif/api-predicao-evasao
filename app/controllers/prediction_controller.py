# app/controllers/prediction_controller.py

import pandas as pd
from typing import List
from app.schemas.prediction_schema import StudentInput, PredictionOutput
from app.services.ml_service import ml_service
from app.core.config import settings

class PredictionController:
    
    def predict(self, student_input: StudentInput) -> PredictionOutput:
        """
        Orquestra a predição para um único aluno.
        """
        # Transforma o Pydantic model em um DataFrame
        input_df = pd.DataFrame([student_input.model_dump()])
        
        # Chama o serviço de ML para obter a probabilidade
        prob = ml_service.predict_proba(input_df)[0]
        
        # Aplica o threshold para definir a classe
        classe_prevista = 1 if prob >= settings.THRESHOLD else 0

        # Monta o objeto de saída
        return PredictionOutput(
            prob_evasao=prob,
            classe_prevista=classe_prevista,
            threshold=settings.THRESHOLD
        )

    def predict_batch(self, students_input: List[StudentInput]) -> List[PredictionOutput]:
        """
        Orquestra a predição para múltiplos alunos (em lote).
        """
        # Converte a lista de Pydantic models em um DataFrame
        input_df = pd.DataFrame([s.model_dump() for s in students_input])

        # Chama o serviço para obter as probabilidades
        probabilities = ml_service.predict_proba(input_df)

        # Monta a lista de objetos de saída
        results = []
        for prob in probabilities:
            classe_prevista = 1 if prob >= settings.THRESHOLD else 0
            results.append(PredictionOutput(
                prob_evasao=prob,
                classe_prevista=classe_prevista,
                threshold=settings.THRESHOLD
            ))
        return results

prediction_controller = PredictionController()