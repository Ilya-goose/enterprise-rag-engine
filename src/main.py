from fastapi import FastAPI
from pydantic import BaseModel
from src.vector_store import VectorStore

# Создаем приложение
app = FastAPI(title="VectorDB Enterprise")

store = VectorStore(dim=3)

# Описываем структуру входящего запроса через Pydantic
class AddRequest(BaseModel):
    vector: list[float]
    payload: dict

class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int
    filters: dict | None


@app.post("/add")
def add(request: AddRequest) -> dict:
    store.add(request.vector, request.payload)
    return {"message": "Vector added successfully"}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    res = store.search(request.query_vector, request.top_k, request.filters)
    return {"results": res}