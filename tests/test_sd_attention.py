import unittest

from vla_attention.teachers.sd_attention import CrossAttentionStore


class TensorStub:
    def __init__(self, shape): self.shape = shape
    def __getitem__(self, item):
        if isinstance(item, slice):
            return TensorStub((self.shape[0] - (item.start or 0), *self.shape[1:]))
        return self
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self


class SDAttentionStoreTests(unittest.TestCase):
    def test_records_only_square_spatial_query_maps(self):
        store = CrossAttentionStore()
        store.record(TensorStub((16, 64, 77)))
        store.record(TensorStub((16, 63, 77)))
        self.assertEqual(len(store.maps[8]), 1)
        self.assertNotIn(7, store.maps)

    def test_only_drops_unconditional_heads_when_declared(self):
        store = CrossAttentionStore(conditional_cfg_half=True)
        store.record(TensorStub((16, 64, 77)))
        self.assertEqual(store.maps[8][0].shape[0], 8)
