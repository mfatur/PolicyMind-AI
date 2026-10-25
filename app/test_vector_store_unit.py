import subprocess
import sys
import textwrap
import types

from app import vector_store


class FakeCollection:
    def __init__(self, name, count=1):
        self.name = name
        self._count = count

    def count(self):
        return self._count


class FakeClient:
    def __init__(self, collections=None):
        self.collections = {
            collection.name: collection
            for collection in collections or []
        }

    def list_collections(self):
        return list(self.collections.values())

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        del self.collections[name]

    def create_collection(self, name, metadata):
        collection = FakeCollection(name, count=0)
        self.collections[name] = collection
        return collection


def install_fake_embedding_model(monkeypatch):
    embedding = types.ModuleType("app.embedding")
    embedding.get_embedding_model = lambda: None
    monkeypatch.setitem(sys.modules, "app.embedding", embedding)


def test_ensure_vector_store_reuses_populated_collection(
    monkeypatch,
    tmp_path,
):
    install_fake_embedding_model(monkeypatch)
    collection = FakeCollection(vector_store.COLLECTION_NAME, count=6)
    client = FakeClient([collection])
    monkeypatch.setattr(vector_store, "DB_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(
        vector_store.chromadb,
        "PersistentClient",
        lambda path: client,
    )
    monkeypatch.setattr(
        vector_store,
        "_build_vector_store",
        lambda _client: (_ for _ in ()).throw(
            AssertionError("Existing vector store should not be rebuilt.")
        ),
    )

    result = vector_store.ensure_vector_store()
    repeated_result = vector_store.ensure_vector_store()

    assert result is collection
    assert repeated_result is collection
    assert result.count() == 6


def test_ensure_vector_store_builds_missing_collection_once(
    monkeypatch,
    tmp_path,
):
    install_fake_embedding_model(monkeypatch)
    client = FakeClient()
    built = FakeCollection(vector_store.COLLECTION_NAME, count=6)
    build_calls = []

    def build_vector_store(received_client):
        build_calls.append(received_client)
        received_client.collections[built.name] = built
        return built

    monkeypatch.setattr(vector_store, "DB_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(
        vector_store.chromadb,
        "PersistentClient",
        lambda path: client,
    )
    monkeypatch.setattr(
        vector_store,
        "_build_vector_store",
        build_vector_store,
    )

    result = vector_store.ensure_vector_store()
    repeated_result = vector_store.ensure_vector_store()

    assert result is built
    assert repeated_result is built
    assert build_calls == [client]


def test_vector_store_lock_serializes_processes(tmp_path):
    worker = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from app import vector_store

        root = Path(sys.argv[1])
        vector_store.DB_DIR = root / "chroma_db"
        with vector_store._vector_store_lock():
            with (root / "events.txt").open("a", encoding="utf-8") as events:
                events.write("enter\\n")
                events.flush()
            time.sleep(0.25)
            with (root / "events.txt").open("a", encoding="utf-8") as events:
                events.write("exit\\n")
                events.flush()
        """
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path)],
        )
        for _ in range(3)
    ]

    try:
        exit_codes = [process.wait(timeout=30) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    events = (tmp_path / "events.txt").read_text().splitlines()

    assert exit_codes == [0, 0, 0]
    assert events == ["enter", "exit"] * 3
