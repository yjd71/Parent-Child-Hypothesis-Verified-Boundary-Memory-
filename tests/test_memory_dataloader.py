import types

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

import data.dataset as dataset_module
from data.dataset import MyData, prepare_labeled_memory_dataloader
from utils.tools import path_to_image


class FakeMemoryDataset(data.Dataset):
    def __init__(self):
        self.image_to_idx = {"img-a": 0, "img-b": 1, "img-c": 2}
        self.image_id_to_indices = {"img-a": [0], "img-b": [1], "img-c": [2]}

    def __len__(self):
        return 3

    def __getitem__(self, index):
        image_id = ("img-a", "img-b", "img-c")[index]
        return torch.full((3, 4, 4), float(index)), torch.zeros(1, 4, 4), image_id, index


def _loader_config():
    return types.SimpleNamespace(
        training_set="fake",
        img_size=4,
        batch_size=2,
        num_workers=0,
    )


def test_memory_loader_deduplicates_ids_and_keeps_tail_batch(monkeypatch):
    fake_dataset = FakeMemoryDataset()
    monkeypatch.setattr(dataset_module, "MyData", lambda **kwargs: fake_dataset)

    loader = prepare_labeled_memory_dataloader(
        _loader_config(),
        ["img-a", "img-a", "img-b", "img-c"],
    )

    batches = list(loader)
    image_ids = [image_id for batch in batches for image_id in batch[2]]
    assert len(loader.dataset) == 3
    assert len(batches) == 2
    assert [batch[0].size(0) for batch in batches] == [2, 1]
    assert image_ids == ["img-a", "img-b", "img-c"]


def test_memory_loader_rejects_ambiguous_dataset_image_ids(monkeypatch):
    fake_dataset = FakeMemoryDataset()
    fake_dataset.image_to_idx["img-a"] = 1
    fake_dataset.image_id_to_indices["img-a"] = [0, 1]
    monkeypatch.setattr(dataset_module, "MyData", lambda **kwargs: fake_dataset)

    with pytest.raises(ValueError, match="Duplicate image IDs"):
        prepare_labeled_memory_dataloader(_loader_config(), ["img-a"])


def test_path_to_image_nearest_resize_preserves_binary_mask(tmp_path):
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    resized = path_to_image(
        str(mask_path),
        size=(7, 7),
        color_type="gray",
        interpolation=cv2.INTER_NEAREST,
    )

    assert set(np.asarray(resized).reshape(-1).tolist()) == {0, 255}


def test_memory_dataset_view_is_deterministic_and_skips_augmentation(monkeypatch, tmp_path):
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[:, :, 1] = 127
    mask = np.array(
        [
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "sample.jpg"
    mask_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    config = types.SimpleNamespace(img_size=7, preproc_methods=["flip", "rotate"])
    memory_dataset = MyData.__new__(MyData)
    memory_dataset.config = config
    memory_dataset.load_all = False
    memory_dataset.image_paths = [str(image_path)]
    memory_dataset.label_paths = [str(mask_path)]
    memory_dataset.is_train = True
    memory_dataset.apply_augmentation = False
    memory_dataset.label_resize_interpolation = cv2.INTER_NEAREST
    memory_dataset.transform_image = transforms.Compose(
        [
            transforms.Resize((7, 7)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    memory_dataset.transform_label = transforms.Compose(
        [
            transforms.Resize((7, 7), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    monkeypatch.setattr(
        dataset_module,
        "preproc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("augmentation must be disabled")),
    )

    first = memory_dataset[0]
    second = memory_dataset[0]

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert set(first[1].unique().tolist()) == {0.0, 1.0}
    assert first[2] == "sample"
