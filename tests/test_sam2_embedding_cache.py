import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from SAM.SAM_refinement.svb_cache import SAMImageEmbeddingCache
from SAM.SAM_refinement.sam_backend_adapter import ExistingSAMBackendAdapter
from SAM.protoSAMprompt.sam2_refiner import Sam2PromptRefiner


def _config(cache_dir, enabled=True, disk=True, store_dtype="float32", version="state-v1"):
    return SimpleNamespace(
        use_sam_embedding_cache=enabled,
        sam_image_embedding_cache_size=64,
        sam2_image_embedding_cache_size=16,
        sam_embedding_cache_disk=disk,
        sam_embedding_cache_dir=str(cache_dir),
        sam_embedding_cache_max_gb=1,
        sam_embedding_cache_prune_interval=1,
        sam_embedding_cache_store_dtype="float16",
        sam2_embedding_cache_store_dtype=store_dtype,
        sam_embedding_cache_version=version,
    )


def _state(value=1.0, dtype=torch.float16, hw=(8, 10)):
    return {
        "image_embed": torch.full((1, 4, 2, 2), value, dtype=dtype),
        "high_res_feats": (
            torch.full((1, 2, 8, 8), value + 1, dtype=dtype),
            torch.full((1, 3, 4, 4), value + 2, dtype=dtype),
        ),
        "orig_hw": (hw,),
    }


class _FakeModel:
    image_size = 1024
    directly_add_no_mem_embed = True


class _FakePredictor:
    def __init__(self):
        self.encoder_calls = 0
        self.reset_predictor()

    def reset_predictor(self):
        self._features = None
        self._orig_hw = None
        self._is_image_set = False
        self._is_batch = False

    def set_image(self, image):
        self.encoder_calls += 1
        value = float(np.asarray(image, dtype=np.float32).sum()) + 1.0
        state = _state(value=value, dtype=torch.float16, hw=image.shape[:2])
        self._features = {
            "image_embed": state["image_embed"],
            "high_res_feats": list(state["high_res_feats"]),
        }
        self._orig_hw = list(state["orig_hw"])
        self._is_image_set = True
        self._is_batch = False

    def predict(self, **kwargs):
        height, width = self._orig_hw[0]
        masks = np.ones((1, height, width), dtype=np.uint8)
        return masks, np.array([0.9], dtype=np.float32), np.ones((1, 256, 256), dtype=np.float32)


def _refiner(cache):
    refiner = object.__new__(Sam2PromptRefiner)
    refiner.checkpoint = Path("fake.pt")
    refiner.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    refiner.device = torch.device("cpu")
    refiner.multimask_output = False
    refiner.use_bfloat16 = False
    refiner.embedding_cache = cache
    refiner.model = _FakeModel()
    refiner.predictor = _FakePredictor()
    return refiner


class SAM2EmbeddingCacheTests(unittest.TestCase):
    def test_sam1_tensor_payload_remains_backward_compatible(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = _config(cache_dir)
            cache = SAMImageEmbeddingCache(cfg, backend_tag="sam1", model_tag="vit_h")
            image = np.zeros((3, 4, 3), dtype=np.uint8)
            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return torch.full((1, 2, 2, 2), 0.125, dtype=torch.float32)

            first, first_hit = cache.get_or_compute(image, compute, dtype=torch.float32)
            second, second_hit = cache.get_or_compute(image, compute, dtype=torch.float32)
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertTrue(torch.is_tensor(first))
            self.assertEqual(first.dtype, torch.float32)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(calls["count"], 1)

    def test_structured_payload_disabled_miss_memory_and_disk_hit_are_fp32(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            compute_calls = {"count": 0}

            def compute():
                compute_calls["count"] += 1
                return _state()

            disabled = SAMImageEmbeddingCache(
                _config(cache_dir, enabled=False), backend_tag="sam2", model_tag="model"
            )
            value, hit = disabled.get_or_compute(image, compute, dtype=torch.float32)
            self.assertFalse(hit)
            self.assertEqual(value["image_embed"].dtype, torch.float32)
            self.assertEqual(disabled.cache_info()["size"], 0)

            cache = SAMImageEmbeddingCache(
                _config(cache_dir), backend_tag="sam2", model_tag="model"
            )
            first, first_hit = cache.get_or_compute(image, compute, dtype=torch.float32)
            second, second_hit = cache.get_or_compute(image, compute, dtype=torch.float32)
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first["image_embed"].dtype, torch.float32)
            self.assertEqual(second["high_res_feats"][0].dtype, torch.float32)
            self.assertEqual(cache.cache_info()["memory_hits"], 1)

            restarted = SAMImageEmbeddingCache(
                _config(cache_dir), backend_tag="sam2", model_tag="model"
            )
            disk_value, disk_hit = restarted.get_or_compute(image, compute, dtype=torch.float32)
            self.assertTrue(disk_hit)
            self.assertEqual(restarted.cache_info()["disk_hits"], 1)
            self.assertTrue(torch.equal(first["image_embed"], disk_value["image_embed"]))
            self.assertEqual(compute_calls["count"], 2)

    def test_exact_image_content_and_cache_tags_invalidate_entries(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            changed = image.copy()
            changed[0, 0, 0] = 1
            cfg = _config(cache_dir)
            cache = SAMImageEmbeddingCache(cfg, backend_tag="sam2", model_tag="checkpoint-a|cfg-a")
            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return _state(value=float(calls["count"]), hw=image.shape[:2])

            cache.get_or_compute(image, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA)
            cache.get_or_compute(image, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA)
            cache.get_or_compute(changed, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA)
            self.assertEqual(calls["count"], 2)

            different_model = SAMImageEmbeddingCache(
                cfg, backend_tag="sam2", model_tag="checkpoint-b|cfg-a"
            )
            different_model.get_or_compute(image, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA)
            different_schema = SAMImageEmbeddingCache(
                cfg, backend_tag="sam2", model_tag="checkpoint-a|cfg-a"
            )
            different_schema.get_or_compute(image, compute, extra_tag="state-v2")
            different_dtype = SAMImageEmbeddingCache(
                _config(cache_dir, store_dtype="float16"),
                backend_tag="sam2",
                model_tag="checkpoint-a|cfg-a",
            )
            different_dtype.get_or_compute(image, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA)
            different_version = SAMImageEmbeddingCache(
                _config(cache_dir, version="state-v2"),
                backend_tag="sam2",
                model_tag="checkpoint-a|cfg-a",
            )
            different_version.get_or_compute(
                image, compute, extra_tag=Sam2PromptRefiner.CACHE_SCHEMA
            )
            self.assertEqual(calls["count"], 6)

    def test_corrupt_disk_entry_is_removed_and_recomputed(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            cfg = _config(cache_dir)
            cache = SAMImageEmbeddingCache(cfg, backend_tag="sam2", model_tag="model")
            cache.get_or_compute(image, lambda: _state(hw=image.shape[:2]), extra_tag="schema")
            path = cache._path_for_key(cache.make_key(image, extra_tag="schema"))
            path.write_bytes(b"not a torch payload")

            restarted = SAMImageEmbeddingCache(cfg, backend_tag="sam2", model_tag="model")
            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return _state(value=7.0, hw=image.shape[:2])

            value, hit = restarted.get_or_compute(
                image,
                compute,
                dtype=torch.float32,
                extra_tag="schema",
                validator=_refiner(restarted)._validate_predictor_state,
            )
            self.assertFalse(hit)
            self.assertEqual(calls["count"], 1)
            self.assertEqual(float(value["image_embed"].mean()), 7.0)
            self.assertGreaterEqual(restarted.cache_info()["invalid_entries"], 1)
            self.assertGreaterEqual(restarted.cache_info()["recomputes"], 1)

    def test_set_image_restores_complete_state_without_reencoding(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = _config(cache_dir)
            image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
            cache = SAMImageEmbeddingCache(cfg, backend_tag="sam2", model_tag="checkpoint|cfg")
            refiner = _refiner(cache)

            first = refiner.set_image(image)
            first_embed = refiner.predictor._features["image_embed"].clone()
            second = refiner.set_image(image)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["cache_source"], "memory")
            self.assertEqual(refiner.predictor.encoder_calls, 1)
            self.assertTrue(refiner.predictor._is_image_set)
            self.assertFalse(refiner.predictor._is_batch)
            self.assertEqual(refiner.predictor._orig_hw, [(8, 10)])
            self.assertEqual(len(refiner.predictor._features["high_res_feats"]), 2)
            self.assertEqual(refiner.predictor._features["image_embed"].dtype, torch.float32)
            self.assertTrue(torch.equal(first_embed.float(), refiner.predictor._features["image_embed"]))

            restarted_cache = SAMImageEmbeddingCache(
                cfg, backend_tag="sam2", model_tag="checkpoint|cfg"
            )
            restarted = _refiner(restarted_cache)
            disk = restarted.set_image(image)
            self.assertTrue(disk["cache_hit"])
            self.assertEqual(disk["cache_source"], "disk")
            self.assertEqual(restarted.predictor.encoder_calls, 0)
            self.assertTrue(torch.equal(first_embed.float(), restarted.predictor._features["image_embed"]))

            changed = image.copy()
            changed[-1, -1, -1] ^= 1
            miss = restarted.set_image(changed)
            self.assertFalse(miss["cache_hit"])
            self.assertEqual(restarted.predictor.encoder_calls, 1)

    def test_legacy_refinement_reuses_the_same_set_image_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = SAMImageEmbeddingCache(
                _config(cache_dir, disk=False), backend_tag="sam2", model_tag="model"
            )
            refiner = _refiner(cache)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            coarse = np.ones((1, 8, 10), dtype=np.uint8)
            first, _ = refiner(
                image,
                coarse,
                use_point=False,
                use_box=True,
                use_mask=False,
                iters=1,
            )
            second, _ = refiner(
                image,
                coarse,
                use_point=False,
                use_box=True,
                use_mask=False,
                iters=1,
            )
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(refiner.predictor.encoder_calls, 1)

    def test_external_prompt_path_uses_cache_and_reports_diagnostics(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = SAMImageEmbeddingCache(
                _config(cache_dir, disk=False), backend_tag="sam2", model_tag="model"
            )
            refiner = _refiner(cache)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            adapter = object.__new__(ExistingSAMBackendAdapter)
            adapter.refiner = SimpleNamespace(sam2_refiner=refiner)
            adapter.multimask_output = False
            adapter._denormalize_batch = lambda images: [image]
            images = torch.zeros((1, 3, 8, 10), dtype=torch.float32)
            teacher = torch.zeros((1, 1, 8, 10), dtype=torch.float32)
            boxes = torch.tensor([[[1.0, 1.0, 6.0, 6.0]]])

            first = adapter._predict_sam2_external_prompt(
                images, teacher, boxes, None, None, None
            )
            second = adapter._predict_sam2_external_prompt(
                images, teacher, boxes, None, None, None
            )
            self.assertEqual(refiner.predictor.encoder_calls, 1)
            self.assertEqual(first["backend_aux"]["embedding_cache_misses"], 1)
            self.assertEqual(second["backend_aux"]["embedding_cache_hits"], 1)
            self.assertEqual(second["backend_aux"]["embedding_dtype"], "torch.float32")
            self.assertEqual(
                second["backend_aux"]["embedding_stats"][0]["cache_source"], "memory"
            )
            self.assertTrue(second["valid_candidates"].all().item())


if __name__ == "__main__":
    unittest.main()
