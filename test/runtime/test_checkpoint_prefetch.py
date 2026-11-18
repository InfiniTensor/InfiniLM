"""Reader byte fidelity, storage ownership and bounded prefetch regressions."""
import ctypes
import gc
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from infinilm import _core as core
from infinilm import modeling_utils as loading
from infinilm import _safetensors_mmap as reader


class CheckpointPrefetchTest(unittest.TestCase):
    def test_reader_modes_and_borrowed_storage(self):
        values = {
            'bf16': torch.tensor([1.0, -2.0, 0.125], dtype=torch.bfloat16),
            'int8': torch.tensor([-128, -1, 0, 127], dtype=torch.int8),
            'scale': torch.tensor([1e-8, 1.25], dtype=torch.float32),
            'scalar': torch.tensor(17, dtype=torch.int64),
            'empty': torch.empty((0, 3), dtype=torch.float32),
        }
        expected = {k: v.reshape(-1).view(torch.uint8).numpy().tobytes() for k, v in values.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.safetensors'
            save_file(values, str(path), metadata={'format': 'pt'})
            for mode, resident in [('mmap', True), ('buffered', False), ('auto', True), ('auto', False)]:
                with self.subTest(mode=mode, resident=resident), patch.object(reader, '_is_fully_resident', return_value=resident):
                    shard = reader.SafeTensorsMmap(path, read_mode=mode)
                    arrays = {k: shard.tensor_bytes(k) for k in values}
                    self.assertEqual(shard.read_mode, ('mmap' if resident else 'buffered') if mode == 'auto' else mode)
                    shard.close()
                    gc.collect()
                    for k, array in arrays.items():
                        self.assertFalse(array.flags.writeable)
                        self.assertEqual(array.tobytes(), expected[k])
                        self.assertEqual(array.shape, tuple(values[k].shape))
                    params = loading._load_checkpoint_shard(str(path), None, (), raw=True, read_mode=mode)
                    gc.collect()
                    for k, value in params.items():
                        self.assertEqual(value.device.type, 'cpu')
                        self.assertEqual(ctypes.string_at(value.data_ptr(), len(expected[k])), expected[k])
                    self.assertEqual(params['bf16'].dtype, core.bfloat16)
                    self.assertEqual(params['int8'].dtype, core.int8)
                    self.assertEqual(params['scale'].dtype, core.float32)
            with patch.object(reader.ctypes, 'CDLL', side_effect=OSError('unavailable')):
                with reader.SafeTensorsMmap(path, read_mode='auto') as shard:
                    self.assertEqual(shard.read_mode, 'buffered')
            with self.assertRaises(ValueError):
                reader.SafeTensorsMmap(path, read_mode='invalid')
            path.write_bytes(b'invalid')
            for mode in ('mmap', 'buffered', 'auto'):
                with self.assertRaises(ValueError):
                    reader.SafeTensorsMmap(path, read_mode=mode)

    def test_qwen3_moe_uses_native_auto_reader_and_prefetch(self):
        values = {
            'model.embed_tokens.weight': torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            'lm_head.weight': torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        }
        expected = {k: v.view(torch.uint8).numpy().tobytes() for k, v in values.items()}
        class Model:
            hf_config = {'model_type': 'qwen3_moe', 'tie_word_embeddings': False}
            distributed_config = None
            def __init__(self):
                self.storage = {k: core.empty([2], dtype=core.bfloat16, device='cpu') for k in values}
                self.loaded = {}
                self.finalized = False
            def state_dict_keyname(self):
                return list(values)
            def state_dict(self):
                return [{k: v._underlying for k, v in self.storage.items()}]
            def load_state_dict(self, params, strict):
                assert not strict
                self.loaded.update({k: ctypes.string_at(v.data_ptr(), 4) for k, v in params.items()})
            def process_weights_after_loading(self):
                self.finalized = True
        reads = []
        original = loading._load_checkpoint_shard
        def observe(*args, **kwargs):
            reads.append((threading.get_ident(), kwargs['read_mode']))
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(Model.hf_config))
            for index, (name, tensor) in enumerate(values.items()):
                save_file({name: tensor}, str(root/f'{index}.safetensors'))
            model = Model()
            with patch.dict(os.environ, {'HYGON_ARCH': 'gfx936'}), patch.object(loading, '_load_checkpoint_shard', side_effect=observe), patch.object(loading, '_require_torch_checkpoint', side_effect=AssertionError('native reader required')):
                os.environ.pop('INFINILM_WEIGHT_READ_MODE', None)
                loading.load_model_state_dict_by_file(model, str(root), core.bfloat16)
            self.assertEqual(model.loaded, expected)
            self.assertTrue(model.finalized)
            self.assertEqual(len(reads), 2)
            self.assertTrue(all(t != threading.get_ident() and mode == 'auto' for t, mode in reads))

    def test_prefetch_one_shard_ahead_and_close(self):
        reads = []
        second_ready = threading.Event()
        def read(path):
            reads.append(path)
            if path == 1:
                second_ready.set()
            return {'value': path}, 0.0
        iterator = loading._prefetch_checkpoint_shards(list(range(4)), read)
        self.assertEqual(next(iterator)[1], {'value': 0})
        self.assertTrue(second_ready.wait(5))
        self.assertEqual(reads, [0, 1])
        iterator.close()
        self.assertEqual(reads, [0, 1])

    def test_prefetch_propagates_reader_error(self):
        def read(path):
            if path == 1:
                raise OSError('broken shard')
            return {}, 0.0
        iterator = loading._prefetch_checkpoint_shards([0, 1, 2], read)
        next(iterator)
        with self.assertRaisesRegex(OSError, 'broken shard'):
            next(iterator)


if __name__ == '__main__':
    unittest.main()
