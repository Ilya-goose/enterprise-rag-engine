from src.vector_store import VectorStore
from numpy import array


def test_vector_store_basic():
    store = VectorStore(dim=3)

    store.add([1.0, 0.0, 0.0], "Книга 'Мой любимый Python'")
    store.add([0.0, 1.0, 0.0], "Как же мне нравятся лиминальные пространства")
    store.add([0.0, 0.0, 1.0], "Криштиану Роналду - история становления")

    query_vector = array([0.8, 0.15, 0.05])

    res = store.search(query_vector, top_k=1)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0] == "Книга 'Мой любимый Python'"


def test_vector_store_k_bigger_len():
    store = VectorStore(dim=3)
    store.add([1.0, 0.0, 0.0], "Книга 'Мой любимый Python'")
    store.add([0.0, 1.0, 0.0], "Как же мне нравятся лиминальные пространства")
    query_vector = array([0.0, 0.9, 0.1])
    res = store.search(query_vector, top_k=3)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] == "Как же мне нравятся лиминальные пространства"


def test_vector_store_empty_base():
    store = VectorStore(dim=3)
    query_vector = array([0.0, 0.9, 0.1])
    res = store.search(query_vector, top_k=3)
    assert isinstance(res, list)
    assert len(res) == 0


def test_vector_store_zero_vector():
    store = VectorStore(dim=3)
    store.add([1.0, 0.0, 0.0], "Книга 'Мой любимый Python'")
    store.add([0.0, 1.0, 0.0], "Как же мне нравятся лиминальные пространства")
    store.add([0.0, 0.0, 1.0], "Криштиану Роналду - история становления")
    query_vector = array([0.0, 0.0, 0.0])
    res = store.search(query_vector, top_k=1)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0] == "Книга 'Мой любимый Python'"




