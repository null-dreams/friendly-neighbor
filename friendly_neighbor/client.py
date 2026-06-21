import os
import sqlite3
import struct
import json
from typing import List, Dict, Any, Optional, Tuple, Union

def pack_vector(floats: List[float]) -> bytes:
    """Serialize a list of floats into a binary BLOB (little-endian float array)."""
    return struct.pack(f"{len(floats)}f", *floats)

def unpack_vector(blob: bytes) -> List[float]:
    """Deserialize a binary BLOB back into a list of floats."""
    num_floats = len(blob) // 4
    return list(struct.unpack(f"{num_floats}f", blob))

class Collection:
    def __init__(self, client: "FriendlyNeighborClient", name: str, dimensions: int):
        self.client = client
        self.name = name
        self.dimensions = dimensions
        self.table_name = f"collection_{name}"

    def insert(self, id: Union[int, str], embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Insert a single vector with optional metadata into the collection."""
        if len(embedding) != self.dimensions:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match collection dimension {self.dimensions}")
        
        meta_str = json.dumps(metadata) if metadata is not None else None
        blob = pack_vector(embedding)
        
        with self.client.conn:
            self.client.conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (id, embedding, metadata) VALUES (?, ?, ?)",
                (str(id), blob, meta_str)
            )

    def insert_many(self, ids: List[Union[int, str]], embeddings: List[List[float]], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None):
        """Insert multiple vectors with optional metadata into the collection."""
        if len(ids) != len(embeddings):
            raise ValueError("Length of ids and embeddings must be equal")
        if metadatas is not None and len(ids) != len(metadatas):
            raise ValueError("Length of ids and metadatas must be equal")

        data = []
        for i, (id_, emb) in enumerate(zip(ids, embeddings)):
            if len(emb) != self.dimensions:
                raise ValueError(f"Embedding at index {i} dimension {len(emb)} does not match collection dimension {self.dimensions}")
            meta = metadatas[i] if metadatas is not None else None
            meta_str = json.dumps(meta) if meta is not None else None
            data.append((str(id_), pack_vector(emb), meta_str))

        with self.client.conn:
            self.client.conn.executemany(
                f"INSERT OR REPLACE INTO {self.table_name} (id, embedding, metadata) VALUES (?, ?, ?)",
                data
            )

    def query(self, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Query the collection for the nearest neighbors of the given vector."""
        if len(vector) != self.dimensions:
            raise ValueError(f"Query vector dimension {len(vector)} does not match collection dimension {self.dimensions}")

        query_blob = pack_vector(vector)
        cursor = self.client.conn.cursor()
        cursor.execute(
            f"SELECT id, embedding, metadata, l2_distance(embedding, ?) AS distance FROM {self.table_name} ORDER BY distance ASC LIMIT ?",
            (query_blob, limit)
        )
        
        results = []
        for row in cursor.fetchall():
            res_id, res_emb_blob, res_meta_str, distance = row
            results.append({
                "id": res_id,
                "embedding": unpack_vector(res_emb_blob),
                "metadata": json.loads(res_meta_str) if res_meta_str is not None else None,
                "distance": distance
            })
        return results

    def get(self, id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Retrieve a single vector by its ID."""
        cursor = self.client.conn.cursor()
        cursor.execute(
            f"SELECT id, embedding, metadata FROM {self.table_name} WHERE id = ?",
            (str(id),)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        res_id, res_emb_blob, res_meta_str = row
        return {
            "id": res_id,
            "embedding": unpack_vector(res_emb_blob),
            "metadata": json.loads(res_meta_str) if res_meta_str is not None else None
        }

    def delete(self, id: Union[int, str]):
        """Delete a vector by its ID."""
        with self.client.conn:
            self.client.conn.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (str(id),))

    def delete_many(self, ids: List[Union[int, str]]):
        """Delete multiple vectors by their IDs."""
        with self.client.conn:
            self.client.conn.executemany(f"DELETE FROM {self.table_name} WHERE id = ?", [(str(id_),) for id_ in ids])

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        cursor = self.client.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]


class FriendlyNeighborClient:
    def __init__(self, db_path: str = ":memory:", extension_path: Optional[str] = None):
        """
        Initialize the client and load the friendly-neighbor SQLite extension.
        """
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)

        if extension_path is None:
            # Try to auto-detect extension path from common build directories
            possible_paths = [
                os.path.abspath("build/friendly_neighbor.so"),
                os.path.abspath("build/friendly_neighbor.dll"),
                os.path.abspath("friendly_neighbor.so"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    extension_path = path
                    break
            
            if extension_path is None:
                raise FileNotFoundError(
                    "SQLite extension 'friendly_neighbor' not found. "
                    "Please compile the C++ code or provide the extension_path parameter explicitly."
                )

        self.conn.load_extension(extension_path)
        self._init_meta_table()

    def _init_meta_table(self):
        """Initialize the metadata table tracking collections."""
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS _collections_metadata ("
                "name TEXT PRIMARY KEY, "
                "dimensions INTEGER"
                ")"
            )

    def create_collection(self, name: str, dimensions: int) -> Collection:
        """Create a new collection with specified dimensions."""
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than 0")

        # Check if collection already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT dimensions FROM _collections_metadata WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row is not None:
            if row[0] != dimensions:
                raise ValueError(f"Collection '{name}' already exists with different dimensions ({row[0]} vs {dimensions})")
            return Collection(self, name, dimensions)

        table_name = f"collection_{name}"
        with self.conn:
            self.conn.execute(
                f"CREATE TABLE {table_name} ("
                f"id TEXT PRIMARY KEY, "
                f"embedding BLOB NOT NULL, "
                f"metadata TEXT"
                f")"
            )
            self.conn.execute(
                "INSERT INTO _collections_metadata (name, dimensions) VALUES (?, ?)",
                (name, dimensions)
            )
        return Collection(self, name, dimensions)

    def get_collection(self, name: str) -> Collection:
        """Retrieve an existing collection by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT dimensions FROM _collections_metadata WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Collection '{name}' does not exist.")
        return Collection(self, name, row[0])

    def list_collections(self) -> List[str]:
        """List names of all collections."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM _collections_metadata")
        return [row[0] for row in cursor.fetchall()]

    def delete_collection(self, name: str):
        """Delete a collection and its associated table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM _collections_metadata WHERE name = ?", (name,))
        if cursor.fetchone() is None:
            raise ValueError(f"Collection '{name}' does not exist.")

        table_name = f"collection_{name}"
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute("DELETE FROM _collections_metadata WHERE name = ?", (name,))

    def close(self):
        """Close the SQLite connection."""
        self.conn.close()
