# API para Predição de Evasão de Alunos

## Sumário

Este repositório contém o código-fonte de uma API RESTful desenvolvida para fornecer predições sobre a evasão de alunos. A aplicação utiliza um modelo de Regressão Logística treinado com a biblioteca Scikit-learn e é servida através do framework web FastAPI. A arquitetura do projeto aplica o padrão Model-View-Controller (MVC) para uma clara separação de responsabilidades e manutenibilidade.

## Tabela de Conteúdos
- [Visão Geral da Arquitetura](#visão-geral-da-arquitetura)
- [Pilha Tecnológica](#pilha-tecnológica)
- [Estrutura do Diretório](#estrutura-do-diretório)
- [Instalação e Execução](#instalação-e-execução)
  - [Pré-requisitos](#pré-requisitos)
  - [Configuração do Ambiente](#configuração-do-ambiente)
  - [Treinamento do Modelo](#treinamento-do-modelo)
  - [Inicialização do Servidor](#inicialização-do-servidor)
- [Documentação e Endpoints da API](#documentação-e-endpoints-da-api)
  - [`GET /api/health`](#get-apihealth)
  - [`POST /api/predict`](#post-apipredict)
  - [`POST /api/predict_batch`](#post-apipredict_batch)
- [Licença](#licença)

## Visão Geral da Arquitetura

O sistema foi projetado com os seguintes componentes e funcionalidades chave:

*   **Modelo de Machine Learning:** Um classificador de Regressão Logística é treinado para prever a probabilidade de um aluno evadir (classe 1) ou permanecer (classe 0).
*   **Interface RESTful:** O modelo treinado é exposto através de uma API que segue os princípios REST.
*   **Padrão MVC:** A lógica da aplicação é segregada em:
    *   **Model:** Camada de dados, schemas Pydantic e lógica de serviço de ML.
    *   **View:** Definição das rotas e endpoints da API via `APIRouter` do FastAPI.
    *   **Controller:** Orquestração das requisições, transformações de dados e lógica de negócio.
*   **Validação de Dados:** As requisições e respostas são estritamente tipadas e validadas em tempo de execução utilizando Pydantic.
*   **Documentação OpenAPI:** A API gera automaticamente documentação compatível com a especificação OpenAPI, acessível via Swagger UI e ReDoc.
*   **Inferência em Lote:** Disponibiliza um endpoint para processar múltiplas predições em uma única requisição, otimizando a performance.

## Pilha Tecnológica

*   **Linguagem:** Python 3.11+
*   **Framework da API:** FastAPI
*   **Servidor ASGI:** Uvicorn
*   **Machine Learning:** Scikit-learn
*   **Manipulação de Dados:** Pandas
*   **Validação de Dados:** Pydantic
*   **Serialização do Modelo:** Joblib

## Estrutura do Diretório

O projeto adota uma estrutura modular para promover a organização do código.
```
/
├── app/                  # Contém toda a lógica da aplicação FastAPI
│   ├── main.py           # Ponto de entrada da aplicação e ciclo de vida
│   ├── core/             # Módulos de configuração centralizados
│   ├── controllers/      # Lógica de negócio e orquestração (Controller)
│   ├── services/         # Camada de serviços de baixo nível (ex: interação com o modelo)
│   ├── schemas/          # Schemas Pydantic para validação de dados de entrada/saída
│   └── views/            # Definição das rotas/endpoints (View)
├── data/                 # Dataset utilizado para treinamento
├── model/                # Modelo de ML serializado (.pkl)
├── train_model.py        # Script para execução do pipeline de treinamento
├── .gitignore            # Especifica arquivos a serem ignorados pelo Git
├── README.md             # Este documento
└── requirements.txt      # Dependências do projeto
```

## Instalação e Execução

### Pré-requisitos
*   Python (versão 3.11 ou superior)
*   Git

### Configuração do Ambiente

1.  **Clonar o repositório:**
    ```bash
    git clone https://github.com/esmmif/api-predicao-evasao.git
    cd api-predicao-evasao
    ```

2.  **Criar e ativar um ambiente virtual:**
    ```bash
    # Criar
    python -m venv .venv

    # Ativar no Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1

    # Ativar no macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instalar as dependências do projeto:**
    ```bash
    pip install -r requirements.txt
    ```

### Treinamento do Modelo
O modelo de Regressão Logística deve ser treinado antes da inicialização da API. O script a seguir executa o pipeline de pré-processamento, treinamento, avaliação e serialização do modelo.

```bash
python train_model.py
```
O artefato resultante, `logistic_model.pkl`, será salvo no diretório `model/`.

### Inicialização do Servidor
Para iniciar a API, execute o servidor Uvicorn a partir da raiz do projeto.

```bash
uvicorn app.main:app --reload
```
O servidor estará disponível em `http://127.0.0.1:8000`. O argumento `--reload` habilita o hot-reloading para desenvolvimento.

## Documentação e Endpoints da API
A documentação interativa da API (Swagger UI) é gerada automaticamente e pode ser acessada em **`http://127.0.0.1:8000/docs`**.

### `GET /api/health`
Verifica o status operacional da API e a disponibilidade do modelo de ML.

*   **Resposta de Sucesso (200 OK):**
    ```json
    {
      "status": "ok",
      "model_loaded": true
    }
    ```

### `POST /api/predict`
Executa uma predição para uma única instância de aluno.

*   **Corpo da Requisição (application/json):**
    ```json
    {
      "idade": 19, "sexo": "F", "tipo_escola_medio": "publica", "nota_enem": 650.5,
      "renda_familiar": 2.0, "trabalha": 1, "horas_trabalho_semana": 30,
      "reprovacoes_1_sem": 2, "bolsista": 0, "distancia_campus_km": 12.3
    }
    ```
*   **Resposta de Sucesso (200 OK):**
    ```json
    {
      "prob_evasao": 0.78,
      "classe_prevista": 1,
      "threshold": 0.5
    }
    ```

### `POST /api/predict_batch`
Executa predições para um lote de instâncias de alunos.

*   **Corpo da Requisição (application/json):**
    ```json
    {
      "alunos": [
        { "idade": 19, "sexo": "F", "tipo_escola_medio": "publica", "nota_enem": 650.5, "renda_familiar": 2.0, "trabalha": 1, "horas_trabalho_semana": 30, "reprovacoes_1_sem": 2, "bolsista": 0, "distancia_campus_km": 12.3 },
        { "idade": 22, "sexo": "M", "tipo_escola_medio": "privada", "nota_enem": 710.0, "renda_familiar": 5.5, "trabalha": 0, "horas_trabalho_semana": 0, "reprovacoes_1_sem": 0, "bolsista": 1, "distancia_campus_km": 5.8 }
      ]
    }
    ```
*   **Resposta de Sucesso (200 OK):**
    ```json
    [
      { "prob_evasao": 0.78, "classe_prevista": 1, "threshold": 0.5 },
      { "prob_evasao": 0.15, "classe_prevista": 0, "threshold": 0.5 }
    ]
    ```

## Notebook no Google Colab para teste

[Colab](https://colab.research.google.com/drive/1NuSrL3MaFiV1oHyHe0geSucV495N-k2C?usp=sharing)

## Licença
Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
