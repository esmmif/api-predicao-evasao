# app/schemas/prediction_schema.py

from pydantic import BaseModel, Field
from typing import Literal

# Schema para a entrada de dados de um único aluno
class StudentInput(BaseModel):
    idade: int = Field(..., example=21, description="Idade do aluno")
    sexo: Literal['M', 'F'] = Field(..., example='F', description="Sexo do aluno")
    tipo_escola_medio: Literal['publica', 'privada'] = Field(..., example='publica', description="Tipo de escola do ensino médio")
    nota_enem: float = Field(..., example=650.5, description="Nota do ENEM")
    renda_familiar: float = Field(..., example=3.5, description="Renda familiar em salários mínimos")
    trabalha: int = Field(..., example=1, description="Se o aluno trabalha (1 para sim, 0 para não)")
    horas_trabalho_semana: int = Field(..., example=20, description="Horas de trabalho por semana")
    reprovacoes_1_sem: int = Field(..., example=0, description="Número de reprovações no 1º semestre")
    bolsista: int = Field(..., example=1, description="Se o aluno é bolsista (1 para sim, 0 para não)")
    distancia_campus_km: float = Field(..., example=15.2, description="Distância da casa ao campus em KM")

# Schema para a saída da predição
class PredictionOutput(BaseModel):
    prob_evasao: float = Field(..., example=0.25, description="Probabilidade prevista de evasão")
    classe_prevista: int = Field(..., example=0, description="Classe prevista (1 para evasão, 0 para permanência)")
    threshold: float = Field(..., example=0.5, description="Limiar de decisão utilizado")

# Schema para a resposta do endpoint /health
class HealthCheckResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True