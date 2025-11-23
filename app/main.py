# app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.views import prediction_view
from app.services.ml_service import ml_service
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lógica que executa na inicialização
    print("Iniciando a API e carregando o modelo...")
    ml_service.load_model()
    yield
    # Lógica que executa no encerramento (opcional)
    print("Encerrando a API...")

app = FastAPI(
    title="API de Predição de Evasão de Alunos",
    description="Uma API para prever o risco de evasão de alunos usando Regressão Logística.",
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Inclui as rotas definidas na view
app.include_router(prediction_view.router, prefix="/api")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Bem-vindo à API de Predição de Evasão! Acesse /docs para a documentação."}