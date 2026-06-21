import unittest
import os
from friendly_neighbor import FriendlyNeighborClient, Collection

class TestFriendlyNeighborSDK(unittest.TestCase):
    def setUp(self):
        # Initialize an in-memory client
        # It will automatically detect build/friendly_neighbor.so
        self.client = FriendlyNeighborClient(db_path=":memory:")

    def tearDown(self):
        self.client.close()

    def test_collection_lifecycle(self):
        # 1. Initially no collections
        self.assertEqual(self.client.list_collections(), [])

        # 2. Create collection
        col = self.client.create_collection("test_col", dimensions=4)
        self.assertEqual(col.name, "test_col")
        self.assertEqual(col.dimensions, 4)
        self.assertEqual(self.client.list_collections(), ["test_col"])

        # 3. Retrieve collection
        retrieved_col = self.client.get_collection("test_col")
        self.assertEqual(retrieved_col.name, "test_col")
        self.assertEqual(retrieved_col.dimensions, 4)

        # 4. Attempting to create duplicate with different dimensions should raise ValueError
        with self.assertRaises(ValueError):
            self.client.create_collection("test_col", dimensions=128)

        # 5. Delete collection
        self.client.delete_collection("test_col")
        self.assertEqual(self.client.list_collections(), [])
        
        with self.assertRaises(ValueError):
            self.client.get_collection("test_col")

    def test_collection_crud_and_metadata(self):
        col = self.client.create_collection("user_embeddings", dimensions=3)
        self.assertEqual(col.count(), 0)

        # Insert item with metadata
        col.insert(
            id="user_1",
            embedding=[1.0, 2.0, 3.0],
            metadata={"name": "Alice", "role": "admin"}
        )
        self.assertEqual(col.count(), 1)

        # Get item
        item = col.get("user_1")
        self.assertIsNotNone(item)
        self.assertEqual(item["id"], "user_1")
        self.assertEqual(item["embedding"], [1.0, 2.0, 3.0])
        self.assertEqual(item["metadata"], {"name": "Alice", "role": "admin"})

        # Update item
        col.insert(
            id="user_1",
            embedding=[1.5, 2.5, 3.5],
            metadata={"name": "Alice", "role": "superuser"}
        )
        item = col.get("user_1")
        self.assertEqual(item["embedding"], [1.5, 2.5, 3.5])
        self.assertEqual(item["metadata"], {"name": "Alice", "role": "superuser"})
        self.assertEqual(col.count(), 1)

        # Delete item
        col.delete("user_1")
        self.assertIsNone(col.get("user_1"))
        self.assertEqual(col.count(), 0)

    def test_bulk_operations(self):
        col = self.client.create_collection("bulk_col", dimensions=2)

        ids = ["id_0", "id_1", "id_2"]
        embeddings = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        metadatas = [
            {"label": "origin"},
            {"label": "middle"},
            {"label": "outer"}
        ]

        col.insert_many(ids, embeddings, metadatas)
        self.assertEqual(col.count(), 3)

        # Check values
        self.assertEqual(col.get("id_0")["embedding"], [0.0, 0.0])
        self.assertEqual(col.get("id_1")["metadata"], {"label": "middle"})

        # Delete many
        col.delete_many(["id_0", "id_2"])
        self.assertEqual(col.count(), 1)
        self.assertIsNone(col.get("id_0"))
        self.assertIsNotNone(col.get("id_1"))
        self.assertIsNone(col.get("id_2"))

    def test_vector_query(self):
        col = self.client.create_collection("search_col", dimensions=3)

        ids = ["v1", "v2", "v3"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        metadatas = [{"tag": "x"}, {"tag": "y"}, {"tag": "z"}]
        col.insert_many(ids, embeddings, metadatas)

        # Query close to v1: [0.9, 0.1, 0.0]
        results = col.query(vector=[0.9, 0.1, 0.0], limit=2)
        
        self.assertEqual(len(results), 2)
        # The closest should be v1
        self.assertEqual(results[0]["id"], "v1")
        self.assertEqual(results[0]["embedding"], [1.0, 0.0, 0.0])
        self.assertEqual(results[0]["metadata"], {"tag": "x"})
        # Distance calculation: (0.9-1.0)^2 + 0.1^2 + 0 = 0.01 + 0.01 = 0.02
        self.assertAlmostEqual(results[0]["distance"], 0.02, places=5)

        # The next closest should be v2
        self.assertEqual(results[1]["id"], "v2")
        # Distance calculation: 0.9^2 + (0.1-1.0)^2 + 0 = 0.81 + 0.81 = 1.62
        self.assertAlmostEqual(results[1]["distance"], 1.62, places=5)

    def test_dimension_validation(self):
        col = self.client.create_collection("val_col", dimensions=4)

        # Invalid dimensions in insert
        with self.assertRaises(ValueError):
            col.insert("err_id", [1.0, 2.0])

        # Invalid dimensions in insert_many
        with self.assertRaises(ValueError):
            col.insert_many(["id_1"], [[1.0, 2.0]])

        # Invalid dimensions in query
        with self.assertRaises(ValueError):
            col.query([1.0, 2.0])

if __name__ == "__main__":
    unittest.main()
