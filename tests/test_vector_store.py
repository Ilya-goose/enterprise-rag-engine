from src.vector_store import VectorStore


TEST_VECTORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
]


TEST_PAYLOADS = [
    {"id": 1, "category": "news", "year": 2023},
    {"id": 2, "category": "article", "year": 2024, "author": "John"},
    {"id": 3, "category": "news", "year": 2024},
    {"id": 4, "category": "video", "author": "John"},
    {"id": 5, "category": "news", "year": 2023, "author": "Jane"},
]

QUERY_VECTOR = [1.0, 0.0, 0.0]


def test_vector_store_no_filters():
    store = VectorStore(dim=3)

    store.add(TEST_VECTORS[0], TEST_PAYLOADS[0])
    store.add(TEST_VECTORS[1], TEST_PAYLOADS[1])
    store.add(TEST_VECTORS[2], TEST_PAYLOADS[2])
    store.add(TEST_VECTORS[3], TEST_PAYLOADS[3])
    store.add(TEST_VECTORS[4], TEST_PAYLOADS[4])

    res = store.search(QUERY_VECTOR, top_k=2)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0]["category"] == "news"


def test_search_single_filter():
    store = VectorStore(dim=3)

    store.add(TEST_VECTORS[0], TEST_PAYLOADS[0])
    store.add(TEST_VECTORS[1], TEST_PAYLOADS[1])
    store.add(TEST_VECTORS[2], TEST_PAYLOADS[2])
    store.add(TEST_VECTORS[3], TEST_PAYLOADS[3])
    store.add(TEST_VECTORS[4], TEST_PAYLOADS[4])

    res = store.search(QUERY_VECTOR, top_k=2, filters={"category": "news"})
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[1]["year"] == 2023


def test_search_multiple_filters():
    store = VectorStore(dim=3)

    store.add(TEST_VECTORS[0], TEST_PAYLOADS[0])
    store.add(TEST_VECTORS[1], TEST_PAYLOADS[1])
    store.add(TEST_VECTORS[2], TEST_PAYLOADS[2])
    store.add(TEST_VECTORS[3], TEST_PAYLOADS[3])
    store.add(TEST_VECTORS[4], TEST_PAYLOADS[4])

    res = store.search(QUERY_VECTOR, top_k=2, filters={"category": "news", "year": 2023})
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[1]["year"] == 2023


def test_search_missing_key():
    store = VectorStore(dim=3)

    store.add(TEST_VECTORS[0], TEST_PAYLOADS[0])
    store.add(TEST_VECTORS[1], TEST_PAYLOADS[1])
    store.add(TEST_VECTORS[2], TEST_PAYLOADS[2])
    store.add(TEST_VECTORS[3], TEST_PAYLOADS[3])
    store.add(TEST_VECTORS[4], TEST_PAYLOADS[4])

    res = store.search(QUERY_VECTOR, top_k=2, filters={"tag": "premium"})
    assert isinstance(res, list)
    assert len(res) == 0


def test_search_filter_fewer_than_top_k():
    store = VectorStore(dim=3)

    store.add(TEST_VECTORS[0], TEST_PAYLOADS[0])
    store.add(TEST_VECTORS[1], TEST_PAYLOADS[1])
    store.add(TEST_VECTORS[2], TEST_PAYLOADS[2])
    store.add(TEST_VECTORS[3], TEST_PAYLOADS[3])
    store.add(TEST_VECTORS[4], TEST_PAYLOADS[4])

    res = store.search(QUERY_VECTOR, top_k=5, filters={"author": "John"})
    assert isinstance(res, list)
    assert len(res) == 2









