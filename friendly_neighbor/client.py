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
        # Internal table names
        self.vec_table = f"vec_{name}"      # Narrow table (vectors only)
        self.payload_table = f"payload_{name}"  # Wide table (metadata payload)

    def insert(self, id: Union[int, str], embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Insert a single vector with optional metadata into the collection using a split layout."""
        if len(embedding) != self.dimensions:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match collection dimension {self.dimensions}")
        
        meta_str = json.dumps(metadata) if metadata is not None else None
        blob = pack_vector(embedding)
        
        # We wrap both inserts in a single transaction to ensure ACID compliance
        with self.client.conn:
            self.client.conn.execute(
                f"INSERT OR REPLACE INTO {self.vec_table} (id, embedding) VALUES (?, ?)",
                (str(id), blob)
            )
            self.client.conn.execute(
                f"INSERT OR REPLACE INTO {self.payload_table} (id, metadata) VALUES (?, ?)",
                (str(id), meta_str)
            )

    def insert_many(self, ids: List[Union[int, str]], embeddings: List[List[float]], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None):
        """Insert multiple vectors with optional metadata using split bulk insertion."""
        if len(ids) != len(embeddings):
            raise ValueError("Length of ids and embeddings must be equal")
        if metadatas is not None and len(ids) != len(metadatas):
            raise ValueError("Length of ids and metadatas must be equal")

        vec_data = []
        payload_data = []
        
        for i, (id_, emb) in enumerate(zip(ids, embeddings)):
            if len(emb) != self.dimensions:
                raise ValueError(f"Embedding at index {i} dimension {len(emb)} does not match collection dimension {self.dimensions}")
            
            meta = metadatas[i] if metadatas is not None else None
            meta_str = json.dumps(meta) if meta is not None else None
            
            str_id = str(id_)
            vec_data.append((str_id, pack_vector(emb)))
            payload_data.append((str_id, meta_str))

        with self.client.conn:
            self.client.conn.executemany(
                f"INSERT OR REPLACE INTO {self.vec_table} (id, embedding) VALUES (?, ?)",
                vec_data
            )
            self.client.conn.executemany(
                f"INSERT OR REPLACE INTO {self.payload_table} (id, metadata) VALUES (?, ?)",
                payload_data
            )

    def query(self, vector: List[float], limit: int = 10, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the collection using an optimized subquery join.
        SQLite scans the narrow 'vec_table' for distance, and then joins with 'payload_table'
        only for the top matches.
        """
        if len(vector) != self.dimensions:
            raise ValueError(f"Query vector dimension {len(vector)} does not match collection dimension {self.dimensions}")

        query_blob = pack_vector(vector)
        cursor = self.client.conn.cursor()
        
        if filter:
            # We must join payload_table in the subquery to filter by metadata before limiting
            where_clauses = []
            params = [query_blob]
            for key, val in filter.items():
                where_clauses.append(f"json_extract(p_sub.metadata, '$.{key}') = ?")
                params.append(val)
            
            where_str = " AND ".join(where_clauses)
            sql = f"""
                SELECT v.id, v.embedding, p.metadata, v.distance
                FROM (
                    SELECT v_sub.id, v_sub.embedding, l2_distance(v_sub.embedding, ?) AS distance 
                    FROM {self.vec_table} v_sub
                    JOIN {self.payload_table} p_sub ON v_sub.id = p_sub.id
                    WHERE {where_str}
                    ORDER BY distance ASC 
                    LIMIT ?
                ) v
                LEFT JOIN {self.payload_table} p ON v.id = p.id
            """
            params.append(limit)
            cursor.execute(sql, tuple(params))
        else:
            # Subquery optimization: Order first on the narrow table, then join the metadata.
            sql = f"""
                SELECT v.id, v.embedding, p.metadata, v.distance
                FROM (
                    SELECT id, embedding, l2_distance(embedding, ?) AS distance 
                    FROM {self.vec_table} 
                    ORDER BY distance ASC 
                    LIMIT ?
                ) v
                LEFT JOIN {self.payload_table} p ON v.id = p.id
            """
            cursor.execute(sql, (query_blob, limit))
        
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
        """Retrieve a single vector and metadata by its ID."""
        cursor = self.client.conn.cursor()
        # Join narrow and wide tables
        sql = f"""
            SELECT v.id, v.embedding, p.metadata 
            FROM {self.vec_table} v
            LEFT JOIN {self.payload_table} p ON v.id = p.id
            WHERE v.id = ?
        """
        cursor.execute(sql, (str(id),))
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
        """Delete a vector and its metadata by ID."""
        with self.client.conn:
            self.client.conn.execute(f"DELETE FROM {self.vec_table} WHERE id = ?", (str(id),))
            self.client.conn.execute(f"DELETE FROM {self.payload_table} WHERE id = ?", (str(id),))

    def delete_many(self, ids: List[Union[int, str]]):
        """Delete multiple vectors and their metadata by IDs."""
        formatted_ids = [(str(id_),) for id_ in ids]
        with self.client.conn:
            self.client.conn.executemany(f"DELETE FROM {self.vec_table} WHERE id = ?", formatted_ids)
            self.client.conn.executemany(f"DELETE FROM {self.payload_table} WHERE id = ?", formatted_ids)

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        cursor = self.client.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.vec_table}")
        return cursor.fetchone()[0]


class FriendlyNeighborClient:
    def __init__(self, db_path: str = ":memory:", extension_path: Optional[str] = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)

        if extension_path is None:
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
                raise FileNotFoundError("SQLite extension 'friendly_neighbor' not found.")

        self.conn.load_extension(extension_path)
        self._init_meta_table()

    def _init_meta_table(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS _collections_metadata ("
                "name TEXT PRIMARY KEY, "
                "dimensions INTEGER"
                ")"
            )

    def create_collection(self, name: str, dimensions: int) -> Collection:
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than 0")

        cursor = self.conn.cursor()
        cursor.execute("SELECT dimensions FROM _collections_metadata WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row is not None:
            if row[0] != dimensions:
                raise ValueError(f"Collection '{name}' already exists with different dimensions")
            return Collection(self, name, dimensions)

        vec_table = f"vec_{name}"
        payload_table = f"payload_{name}"
        
        with self.conn:
            # Create Narrow Vector Table
            self.conn.execute(
                f"CREATE TABLE {vec_table} ("
                f"id TEXT PRIMARY KEY, "
                f"embedding BLOB NOT NULL"
                f")"
            )
            # Create Wide Payload Table
            self.conn.execute(
                f"CREATE TABLE {payload_table} ("
                f"id TEXT PRIMARY KEY, "
                f"metadata TEXT"
                f")"
            )
            self.conn.execute(
                "INSERT INTO _collections_metadata (name, dimensions) VALUES (?, ?)",
                (name, dimensions)
            )
        return Collection(self, name, dimensions)

    def get_collection(self, name: str) -> Collection:
        cursor = self.conn.cursor()
        cursor.execute("SELECT dimensions FROM _collections_metadata WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Collection '{name}' does not exist.")
        return Collection(self, name, row[0])

    def list_collections(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM _collections_metadata")
        return [row[0] for row in cursor.fetchall()]

    def delete_collection(self, name: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM _collections_metadata WHERE name = ?", (name,))
        if cursor.fetchone() is None:
            raise ValueError(f"Collection '{name}' does not exist.")

        vec_table = f"vec_{name}"
        payload_table = f"payload_{name}"
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {vec_table}")
            self.conn.execute(f"DROP TABLE IF EXISTS {payload_table}")
            self.conn.execute("DELETE FROM _collections_metadata WHERE name = ?", (name,))

    def close(self):
        self.conn.close()