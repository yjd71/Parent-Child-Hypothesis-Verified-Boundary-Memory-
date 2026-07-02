import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, SequentialSampler

with patch("utils.Logger", return_value=SimpleNamespace()):
    from data.dataset import prepare_dataloader


class _IndexDataset(Dataset):
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.long)


class EvaluationDataLoaderTests(unittest.TestCase):
    def test_evaluation_keeps_order_and_incomplete_final_batch(self):
        loader = prepare_dataloader(
            _IndexDataset(2026),
            batch_size=6,
            num_workers=0,
            to_be_distributed=False,
            is_train=False,
        )
        batches = list(loader)
        flattened = torch.cat(batches).tolist()

        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertFalse(loader.drop_last)
        self.assertEqual(len(loader), 338)
        self.assertEqual(len(batches[-1]), 4)
        self.assertEqual(flattened, list(range(2026)))


if __name__ == "__main__":
    unittest.main()
