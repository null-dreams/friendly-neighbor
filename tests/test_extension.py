import sqlite3
import struct
import unittest
import os

def pack_vector(floats):
    return struct.pack(f"{len(floats)}f", *floats)

class TestL2DistanceExtension(unittest.TestCase):
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

    def test_l2_distance_correctness(self):
        # (0, 0, 0) and (3, 4, 0) => L2 squared distance should be 3^2 + 4^2 = 25
        v1 = pack_vector([0.0, 0.0, 0.0])
        v2 = pack_vector([3.0, 4.0, 0.0])
        cursor = self.conn.cursor()
        cursor.execute("SELECT l2_distance(?, ?)", (v1, v2))
        res = cursor.fetchone()[0]
        self.assertAlmostEqual(res, 25.0)

        # Identical vectors => distance should be 0.0
        v3 = pack_vector([1.2, -3.4, 5.6, 7.8])
        cursor.execute("SELECT l2_distance(?, ?)", (v3, v3))
        res = cursor.fetchone()[0]
        self.assertAlmostEqual(res, 0.0)

    def test_invalid_argument_count(self):
        # Too few arguments
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance(?)", (pack_vector([1.0]),))
        self.assertIn("wrong number of arguments to function l2_distance()", str(ctx.exception))

        # Too many arguments
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance(?, ?, ?)", (pack_vector([1.0]), pack_vector([1.0]), pack_vector([1.0])))
        self.assertIn("wrong number of arguments to function l2_distance()", str(ctx.exception))

    def test_invalid_argument_types(self):
        # Non-BLOB arguments (e.g. text)
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance('not a blob', ?)", (pack_vector([1.0]),))
        self.assertIn("Both arguments to l2_distance must be BLOBs", str(ctx.exception))

    def test_mismatched_dimensions(self):
        v1 = pack_vector([1.0, 2.0])
        v2 = pack_vector([1.0, 2.0, 3.0])
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance(?, ?)", (v1, v2))
        self.assertIn("BLOBs must be of equal length", str(ctx.exception))

    def test_empty_vectors(self):
        v1 = pack_vector([])
        v2 = pack_vector([])
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance(?, ?)", (v1, v2))
        self.assertIn("Vector cannot be empty", str(ctx.exception))

    def test_non_float_multiple_size(self):
        v1 = b'\x00' * 5
        v2 = b'\x00' * 5
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute("SELECT l2_distance(?, ?)", (v1, v2))
        self.assertIn("BLOB size must be a multiple of sizeof(float)", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
