# app/views/prediction_view.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict

from app.schemas.prediction_schema import StudentInput, PredictionOutput, HealthCheckResponse
from app.controllers.prediction_controller import prediction_controller
from app.services.ml_service import ml_service

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Endpoint para verificar a saúde da API e se o modelo foi carregado.
    """
    return HealthCheckResponse(model_loaded=ml_service.is_model_loaded())

@router.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_single_student(student_input: StudentInput):
    """
    Prevê a probabilidade de evasão para um único aluno.
    """
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    try:
        result = prediction_controller.predict(student_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao prever: {e}")

@router.post("/predict_batch", response_model=List[PredictionOutput], tags=["Prediction"])
async def predict_multiple_students(request_body: Dict[str, List[StudentInput]]):
    """
    Prevê a probabilidade de evasão para uma lista de alunos.
    """
    students_input = request_body.get("alunos")
    if not students_input:
        raise HTTPException(status_code=422, detail="Corpo da requisição inválido. Esperado: {'alunos': [...]}")

    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
        
    try:
        results = prediction_controller.predict_batch(students_input)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao prever em lote: {e}")