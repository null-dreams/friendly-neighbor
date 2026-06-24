import sqlite3
import struct
import unittest
import os
import random

def pack_vector(floats):
    return struct.pack(f"{len(floats)}f", *floats)

def unpack_vector(blob):
    num_floats = len(blob) // 4
    return list(struct.unpack(f"{num_floats}f", blob))

def l2_distance_squared_py(v1, v2):
    return sum((x - y) ** 2 for x, y in zip(v1, v2))

class TestVectorRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = ":memory:"
        cls.ext_path = os.path.abspath("build/friendly_neighbor.so")
        
        cls.conn = sqlite3.connect(cls.db_path)
        cls.conn.enable_load_extension(True)
        cls.conn.load_extension(cls.ext_path)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def setUp(self):
        # Create a fresh table for each test
        self.conn.execute("DROP TABLE IF EXISTS vectors")
        self.conn.execute("CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding BLOB, metadata TEXT)")

    def test_basic_knn_retrieval(self):
        # Populate table with some reference vectors (3-dimensional)
        vectors = {
            1: [0.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 2.0, 0.0],
            4: [3.0, 4.0, 0.0]
        }
        
        for vid, vec in vectors.items():
            self.conn.execute("INSERT INTO vectors (id, embedding) VALUES (?, ?)", (vid, pack_vector(vec)))
        self.conn.commit()

        # Query vector: [0.5, 0.0, 0.0]
        query = [0.5, 0.0, 0.0]
        query_blob = pack_vector(query)

        # Expected distances squared:
        # id 1: (0.5-0)^2 + 0 + 0 = 0.25
        # id 2: (0.5-1)^2 + 0 + 0 = 0.25
        # id 3: 0.25 + 4.0 = 4.25
        # id 4: (0.5-3)^2 + 16 = 6.25 + 16 = 22.25

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, l2_distance(embedding, ?) AS dist FROM vectors ORDER BY dist ASC", (query_blob,))
        results = cursor.fetchall()

        # Expected ordering: IDs 1 and 2 first (distance 0.25), then 3, then 4
        self.assertEqual(len(results), 4)
        
        # Check closest items
        closest_ids = [results[0][0], results[1][0]]
        self.assertIn(1, closest_ids)
        self.assertIn(2, closest_ids)
        self.assertAlmostEqual(results[0][1], 0.25)
        self.assertAlmostEqual(results[1][1], 0.25)

        # Check next closest
        self.assertEqual(results[2][0], 3)
        self.assertAlmostEqual(results[2][1], 4.25)

        # Check farthest
        self.assertEqual(results[3][0], 4)
        self.assertAlmostEqual(results[3][1], 22.25)

    def test_knn_with_limit(self):
        # Insert 10 vectors: [i, 0, 0] for i = 0..9
        for i in range(10):
            self.conn.execute("INSERT INTO vectors (id, embedding) VALUES (?, ?)", (i, pack_vector([float(i), 0.0, 0.0])))
        self.conn.commit()

        # Query vector: [4.2, 0.0, 0.0], retrieve top 3 nearest
        query_blob = pack_vector([4.2, 0.0, 0.0])
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, l2_distance(embedding, ?) AS dist FROM vectors ORDER BY dist ASC LIMIT 3", (query_blob,))
        results = cursor.fetchall()

        # Expected nearest are:
        # id 4: distance = 0.04
        # id 5: distance = 0.64
        # id 3: distance = 1.44
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 4)
        self.assertAlmostEqual(results[0][1], 0.04, places=5)
        self.assertEqual(results[1][0], 5)
        self.assertAlmostEqual(results[1][1], 0.64, places=5)
        self.assertEqual(results[2][0], 3)
        self.assertAlmostEqual(results[2][1], 1.44, places=5)

    def test_large_scale_random_vectors(self):
        # Set seed for reproducibility
        random.seed(42)
        
        dimensions = 128
        num_vectors = 100
        
        dataset = []
        for i in range(num_vectors):
            vec = [random.uniform(-10.0, 10.0) for _ in range(dimensions)]
            dataset.append(vec)
            self.conn.execute("INSERT INTO vectors (id, embedding) VALUES (?, ?)", (i, pack_vector(vec)))
        self.conn.commit()

        # Query vector
        query = [random.uniform(-10.0, 10.0) for _ in range(dimensions)]
        query_blob = pack_vector(query)

        # Compute exact distances and sort in Python
        py_results = []
        for idx, vec in enumerate(dataset):
            dist = l2_distance_squared_py(vec, query)
            py_results.append((idx, dist))
        py_results.sort(key=lambda x: x[1])

        # Query using SQLite extension
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, l2_distance(embedding, ?) AS dist FROM vectors ORDER BY dist ASC LIMIT 10", (query_blob,))
        db_results = cursor.fetchall()

        # Verify top-10 retrieval results match Python exactly
        self.assertEqual(len(db_results), 10)
        for i in range(10):
            expected_id, expected_dist = py_results[i]
            actual_id, actual_dist = db_results[i]
            self.assertEqual(actual_id, expected_id)
            self.assertAlmostEqual(actual_dist, expected_dist, places=2)

    def test_empty_table_retrieval(self):
        # Database table is empty
        query_blob = pack_vector([1.0, 2.0, 3.0])
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, l2_distance(embedding, ?) FROM vectors", (query_blob,))
        results = cursor.fetchall()
        self.assertEqual(len(results), 0)

    def test_single_vector_retrieval(self):
        # Database table has only 1 vector
        self.conn.execute("INSERT INTO vectors (id, embedding) VALUES (42, ?)", (pack_vector([1.5, 2.5]),))
        self.conn.commit()

        query_blob = pack_vector([1.5, 2.5])
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, l2_distance(embedding, ?) FROM vectors", (query_blob,))
        results = cursor.fetchall()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 42)
        self.assertAlmostEqual(results[0][1], 0.0)

    def test_mismatched_dimension_error_during_query(self):
        # Insert a 3D vector
        self.conn.execute("INSERT INTO vectors (id, embedding) VALUES (1, ?)", (pack_vector([1.0, 2.0, 3.0]),))
        self.conn.commit()

        # Query with a 2D vector
        query_blob = pack_vector([1.0, 2.0])
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT id, l2_distance(embedding, ?) FROM vectors", (query_blob,)).fetchall()
        self.assertIn("BLOBs must be of equal length", str(ctx.exception))

    def test_metadata_filtering(self):
        import json
        # Populate table with vectors (3-dimensional) and JSON metadata
        # One vector has NULL metadata to test null handling
        vectors = [
            (1, [1.0, 0.0, 0.0], {"category": "A", "status": "active", "priority": 1}),
            (2, [0.0, 1.0, 0.0], {"category": "B", "status": "active", "priority": 2}),
            (3, [0.0, 0.0, 1.0], {"category": "A", "status": "inactive", "priority": 1}),
            (4, [1.1, 0.0, 0.0], {"category": "A", "status": "active", "priority": 3}),
            (5, [0.0, 0.0, 0.0], None)
        ]
        
        for vid, vec, meta in vectors:
            meta_val = json.dumps(meta) if meta is not None else None
            self.conn.execute(
                "INSERT INTO vectors (id, embedding, metadata) VALUES (?, ?, ?)",
                (vid, pack_vector(vec), meta_val)
            )
        self.conn.commit()

        query_blob = pack_vector([1.0, 0.0, 0.0])

        # 1. Test filtering by single key
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, l2_distance(embedding, ?) AS dist FROM vectors "
            "WHERE json_extract(metadata, '$.category') = ? "
            "ORDER BY dist ASC",
            (query_blob, "A")
        )
        results = cursor.fetchall()
        # Should return IDs: 1 (dist=0.0), 4 (dist=0.01), 3 (dist=2.0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 4)
        self.assertEqual(results[2][0], 3)
        self.assertAlmostEqual(results[0][1], 0.0)
        self.assertAlmostEqual(results[1][1], 0.01)
        self.assertAlmostEqual(results[2][1], 2.0)

        # 2. Test filtering by multiple keys using AND
        cursor.execute(
            "SELECT id, l2_distance(embedding, ?) AS dist FROM vectors "
            "WHERE json_extract(metadata, '$.category') = ? AND json_extract(metadata, '$.status') = ? "
            "ORDER BY dist ASC",
            (query_blob, "A", "active")
        )
        results = cursor.fetchall()
        # Should return IDs: 1 (dist=0.0), 4 (dist=0.01)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 4)

        # 3. Test filtering with integer values
        cursor.execute(
            "SELECT id, l2_distance(embedding, ?) AS dist FROM vectors "
            "WHERE json_extract(metadata, '$.priority') = ? "
            "ORDER BY dist ASC",
            (query_blob, 1)
        )
        results = cursor.fetchall()
        # Should return IDs: 1 (dist=0.0), 3 (dist=2.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 3)

        # 4. Test filtering with no matches
        cursor.execute(
            "SELECT id FROM vectors "
            "WHERE json_extract(metadata, '$.category') = ?",
            ("C",)
        )
        results = cursor.fetchall()
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
