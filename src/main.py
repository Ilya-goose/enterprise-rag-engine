from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from src.vector_store import VectorStore
from src.config import settings
from src.embedder import Embedder

# Инициализируем компоненты, используя переменные из конфигурации
app = FastAPI(title="VectorDB Enterprise", version="1.0.0")
store = VectorStore(dim=settings.vector_dim)
embedder = Embedder(model_name=settings.embedder_model)


# Описываем структуру входящего запроса через Pydantic
class AddRequest(BaseModel):
    vector: list[float]
    payload: dict

class UpdateRequest(BaseModel):
    vector: list[float] | None = None
    payload: dict | None = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int
    filters: dict | None
    threshold: float | None = 0.0


class FileRequest(BaseModel):
    filepath: str

class BatchAddRequest(BaseModel):
    vectors: list[list[float]]
    payloads: list[dict]

class BatchDeleteRequest(BaseModel):
    list_doc_id: list[str]


class AddTextRequest(BaseModel):
    text: str
    payload: dict


# Обновляем проверку токена на использование settings
def verify_token(x_api_key: str = Header(...)):
    if x_api_key != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.post("/add", dependencies=[Depends(verify_token)])
def add(request: AddRequest) -> dict:
    try:
        store.add(request.vector, request.payload)
        return {"message": "Vector added successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search", dependencies=[Depends(verify_token)])
def search(request: SearchRequest) -> dict:
    try:
        res = store.search(request.query_vector, request.top_k, request.filters, request.threshold)
        return {"results": res}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/save", dependencies=[Depends(verify_token)])
def save(request: FileRequest) -> dict:
    store.save(request.filepath)
    return {"message": f"Database saved to {request.filepath}"}


@app.post("/load", dependencies=[Depends(verify_token)])
def load(request: FileRequest) -> dict:
    global store
    store = VectorStore.load(request.filepath)
    return {"message": f"Database loaded from {request.filepath}"}


@app.post("/add_batch", dependencies=[Depends(verify_token)])
def add_batch(request: BatchAddRequest):
    try:
        store.add_batch(request.vectors, request.payloads)
        return {"message": f"Successfully added {len(request.vectors)} vectors"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/delete_batch", dependencies=[Depends(verify_token)])
def delete_batch(request: BatchDeleteRequest):
    try:
        result = store.delete_batch(request.list_doc_id)
        return {"message": f"Successfully deleted {len(request.list_doc_id) - len(result)} documents",
                "not found": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats", dependencies=[Depends(verify_token)])
def get_stats():
    return store.get_stats()


@app.delete("/delete/{doc_id}", dependencies=[Depends(verify_token)])
def delete_document(doc_id: str) -> dict:
    success = store.delete(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} deleted successfully"}


@app.post("/vacuum", dependencies=[Depends(verify_token)])
def vacuum():
    count_deleted = store.vacuum()
    return {"message": "Vacuum completed", "freed_elements": count_deleted}


@app.put("/update/{doc_id}", dependencies=[Depends(verify_token)])
def update_document(doc_id: str, request: UpdateRequest) -> dict:
    success = store.update(doc_id, request.vector, request.payload)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document updated successfully"}


@app.get("/get/{doc_id}", dependencies=[Depends(verify_token)])
def get(doc_id: str) -> dict | None:
    res = store.get(doc_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"result": res}


@app.post("/add_text", dependencies=[Depends(verify_token)])
def add_text(request: AddTextRequest):
    """
    Эндпоинт "Всё включено": клиент шлет просто текст,
    а движок сам превращает его в вектор и кладет в базу.
    """
    try:
        # 1. Прогоняем текст через нейросеть
        vector = embedder.encode(request.text)

        # 2. Сохраняем оригинальный текст в метаданные, чтобы потом его найти
        payload = request.payload.copy()
        payload["original_text"] = request.text

        # 3. Добавляем в базу
        store.add(vector, payload)
        return {"message": "Text embedded and added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class SearchTextRequest(BaseModel):
    query_text: str
    top_k: int = 3
    filters: dict | None = None
    threshold: float | None = 0.0


@app.post("/search_text", dependencies=[Depends(verify_token)])
def search_text(request: SearchTextRequest):
    """
    Умный поиск по смыслу: клиент передает обычный вопрос/текст,
    нейросеть переводит его в вектор и ищет релевантные совпадения.
    """
    try:
        # 1. Векторизуем поисковый запрос пользователя
        query_vector = embedder.encode(request.query_text)

        # 2. Ищем по векторному пространству
        results = store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filters=request.filters,
            threshold=request.threshold,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))