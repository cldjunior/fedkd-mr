import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('bloco5', Path(__file__).with_name('preparar_bloco5.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class TestParticionamento(unittest.TestCase):
    def test_disjunto_imutavel_e_tamanho(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); source = d/'orig'; source.mkdir(); dest=d/'novos'
            for client in range(1, 4):
                rows=[]
                for i in range(400):
                    rows.append(struct.pack('<96fB', *([float(client*1000+i)]*96), i%4+1))
                (source/f'dataset_priv_esp32_c{client}.bin').write_bytes(b''.join(rows))
            pub_rows=[]
            for client in range(1, 4):
                raw=(source/f'dataset_priv_esp32_c{client}.bin').read_bytes()
                pub_rows.extend(raw[i*385:(i+1)*385] for i in range(0, 20, 3))
            (source/'dataset_pub.bin').write_bytes(b''.join(pub_rows))
            originals={p.name:p.read_bytes() for p in source.iterdir()}
            b.run(source,dest,0.2,42)
            self.assertEqual(originals,{p.name:p.read_bytes() for p in source.iterdir()})
            pub={b.feat(r) for r in b.load(dest/'dataset_pub.bin')}
            train=set(); test=set()
            for client in range(1,4):
                train|={b.feat(r) for r in b.load(dest/f'dataset_train_esp32_c{client}.bin')}
                test|={b.feat(r) for r in b.load(dest/f'dataset_test_esp32_c{client}.bin')}
            self.assertFalse(pub & train)
            self.assertFalse(pub & test)
            self.assertFalse(train & test)
            with self.assertRaises(FileExistsError): b.run(source,dest,0.2,42)

if __name__=='__main__': unittest.main()
