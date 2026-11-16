"""Testes basicos de envelope binario C/Python, sem rede e sem ESP32."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'server'))
from fedkd_wire import pack, unpack, HEADER_LEN, MAGIC, experiment_hash
import unittest
class WireTests(unittest.TestCase):
    def test_round_trip(self):
        data=bytes(range(80))
        self.assertEqual(unpack(pack(data,3,'fedkd-bloco6-test-01'),3,'fedkd-bloco6-test-01'),data)
        self.assertEqual(len(pack(data,3,'fedkd-bloco6-test-01')),80+HEADER_LEN)
    def test_reject_wrong_round(self):
        with self.assertRaises(ValueError):unpack(pack(b'1234',2,'abc'),3,'abc')
    def test_reject_wrong_experiment(self):
        with self.assertRaises(ValueError):unpack(pack(b'1234',2,'abc'),2,'outro')
    def test_reject_old_wire(self):
        with self.assertRaises(ValueError):unpack(bytes(320),2,'abc')
    def test_hash_known_fnv(self):
        self.assertEqual(experiment_hash(''),0xcbf29ce484222325)
        self.assertEqual(experiment_hash('a'),0xaf63dc4c8601ec8c)
    def test_reject_empty_and_round_zero(self):
        with self.assertRaises(ValueError):pack(b'',0,'abc')
        with self.assertRaises(ValueError):pack(b'',1,'')
if __name__=='__main__':unittest.main()
