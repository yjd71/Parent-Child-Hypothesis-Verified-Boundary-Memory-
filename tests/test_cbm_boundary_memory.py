import torch

from CBM import apply_cbm_defaults, build_cbm_pfi
from CBM.boundary.query import build_pred_boundary
from CBM.boundary.regions import build_gt_regions
from CBM.memory.bank import DenseBoundaryMemory


def _square_gt(batch_size=2, height=16, width=16):
    gt = torch.zeros(batch_size, 1, height, width)
    gt[:, :, 4:12, 5:13] = 1.0
    return gt


def test_build_gt_regions_shapes_ranges_and_no_nan():
    gt = _square_gt(height=20, width=20)
    target_size = (10, 12)

    regions = build_gt_regions(gt, target_size)

    for name in ("fg_core", "fg_boundary", "bg_near", "bg_far", "sdf_approx"):
        tensor = regions[name]
        assert tensor.shape == (2, 1, *target_size)
        assert torch.isfinite(tensor).all()

    for name in ("fg_core", "fg_boundary", "bg_near", "bg_far"):
        tensor = regions[name]
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    assert regions["region_label"].shape == (2, *target_size)
    assert regions["region_label"].dtype == torch.long
    assert set(regions["region_label"].unique().tolist()).issubset({0, 1, 2, 3})
    assert regions["sdf_approx"].min().item() >= -1.0
    assert regions["sdf_approx"].max().item() <= 1.0


def test_build_pred_boundary_shapes_ranges_and_no_nan():
    prob = torch.zeros(2, 1, 16, 16)
    prob[:, :, 4:12, 4:12] = 0.9
    prob[:, :, 6:10, 6:10] = 0.55

    b_query, boundary_mask = build_pred_boundary(prob)

    assert b_query.shape == prob.shape
    assert boundary_mask.shape == prob.shape
    assert boundary_mask.dtype == torch.bool
    assert torch.isfinite(b_query).all()
    assert b_query.min().item() >= 0.0
    assert b_query.max().item() <= 1.0
    assert boundary_mask.any()


def test_empty_dense_boundary_memory_returns_safe_empty_tensors():
    memory = DenseBoundaryMemory(print_diagnostics=False)

    image_keys, image_ids = memory.get_image_keys()
    keys, values, meta = memory.get_sub_memory()

    assert not memory.is_ready()
    assert image_keys.shape == (0, 128)
    assert image_ids == []
    assert keys.shape == (0, 128)
    assert values.shape == (0, 8)
    assert meta == []


def test_dense_boundary_memory_append_finalize_and_sub_memory():
    torch.manual_seed(3)
    memory = DenseBoundaryMemory(
        sample_per_image={"fg_core": 2, "fg_boundary": 3, "bg_near": 3, "bg_far": 2},
        max_sizes={"fg_core": 16, "fg_boundary": 16, "bg_near": 16, "bg_far": 16},
        print_diagnostics=False,
    )
    x3 = torch.randn(2, 256, 4, 4)
    p3 = torch.randn(2, 64, 8, 8)
    gt = _square_gt(batch_size=2, height=16, width=16)

    memory.append_batch(x3=x3, p3=p3, gt=gt, img_ids=["img-a", "img-b"])
    memory.finalize()

    assert memory.is_ready()
    image_keys, image_ids = memory.get_image_keys()
    assert image_keys.shape == (2, 128)
    assert image_keys.dtype == torch.float32
    assert image_ids == ["img-a", "img-b"]

    keys, values, meta = memory.get_sub_memory()
    assert keys.shape[1] == 128
    assert values.shape[1] == 8
    assert len(meta) == keys.shape[0]
    assert torch.isfinite(keys).all()
    assert torch.isfinite(values).all()
    assert values[:, :4].min().item() >= 0.0
    assert values[:, :4].max().item() <= 1.0
    assert values[:, 4:6].min().item() >= 0.0
    assert values[:, 4:6].max().item() <= 1.0
    assert values[:, 6].min().item() >= -1.0
    assert values[:, 6].max().item() <= 1.0
    assert values[:, 7].min().item() >= 0.0
    assert values[:, 7].max().item() <= 1.0

    for region, limit in memory.sample_per_image.items():
        assert memory.keys[region].shape[0] <= limit * 2
        assert memory.keys[region].shape[0] <= memory.max_sizes[region]

    idx_keys, idx_values, idx_meta = memory.get_sub_memory(top_img_ids=torch.tensor([0]))
    id_keys, id_values, id_meta = memory.get_sub_memory(top_img_ids=["img-a"])
    assert idx_keys.shape == id_keys.shape
    assert idx_values.shape == id_values.shape
    assert len(idx_meta) == len(id_meta)
    assert all(item["image_id"] == "img-a" for item in idx_meta)


def test_dense_boundary_memory_device_dtype_cpu():
    memory = DenseBoundaryMemory(print_diagnostics=False)
    x3 = torch.randn(1, 32, 2, 2, dtype=torch.float32)
    p3 = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    gt = _square_gt(batch_size=1, height=8, width=8)

    memory.append_batch(x3=x3, p3=p3, gt=gt, img_ids=["cpu"])
    memory.finalize(device=torch.device("cpu"), dtype=torch.float32)
    keys, values, _ = memory.get_sub_memory(device=torch.device("cpu"), dtype=torch.float32)

    assert keys.device.type == "cpu"
    assert values.device.type == "cpu"
    assert keys.dtype == torch.float32
    assert values.dtype == torch.float32


def test_dense_boundary_memory_state_dict_roundtrip():
    memory = DenseBoundaryMemory(
        sample_per_image={"fg_core": 2, "fg_boundary": 2, "bg_near": 2, "bg_far": 2},
        max_sizes={"fg_core": 16, "fg_boundary": 16, "bg_near": 16, "bg_far": 16},
        print_diagnostics=False,
    )
    x3 = torch.randn(2, 32, 2, 2)
    p3 = torch.randn(2, 16, 4, 4)
    gt = _square_gt(batch_size=2, height=8, width=8)
    memory.append_batch(x3=x3, p3=p3, gt=gt, img_ids=["img-a", "img-b"])
    memory.finalize(device=torch.device("cpu"), dtype=torch.float32)

    restored = DenseBoundaryMemory(
        sample_per_image=memory.sample_per_image,
        max_sizes=memory.max_sizes,
        print_diagnostics=False,
    )
    restored.load_state_dict(memory.to_state_dict(), device=torch.device("cpu"), dtype=torch.float32)

    assert restored.is_ready()
    assert restored.get_image_keys()[1] == ["img-a", "img-b"]
    keys, values, meta = restored.get_sub_memory(top_img_ids=["img-b"])
    assert keys.shape[0] == values.shape[0] == len(meta)
    assert keys.shape[1] == 128
    assert values.shape[1] == 8
    assert all(item["image_id"] == "img-b" for item in meta)
    assert torch.isfinite(keys).all()
    assert torch.isfinite(values).all()


def test_dense_boundary_memory_cuda_smoke_if_available():
    if not torch.cuda.is_available():
        return
    memory = DenseBoundaryMemory(print_diagnostics=False)
    x3 = torch.randn(1, 32, 2, 2, device="cuda")
    p3 = torch.randn(1, 16, 4, 4, device="cuda")
    gt = _square_gt(batch_size=1, height=8, width=8).cuda()

    memory.append_batch(x3=x3, p3=p3, gt=gt, img_ids=["cuda"])
    memory.finalize(device=torch.device("cuda"), dtype=torch.float32)
    keys, values, _ = memory.get_sub_memory(device=torch.device("cuda"), dtype=torch.float32)

    assert keys.device.type == "cuda"
    assert values.device.type == "cuda"
    assert torch.isfinite(keys).all()
    assert torch.isfinite(values).all()


def test_cbm_api_defaults_and_engine_smoke():
    class Config:
        cbm_pfi_enable = False
        cbm_top_img_k = 99

    config = Config()
    apply_cbm_defaults(config)
    assert config.cbm_top_img_k == 99
    assert config.cbm_memory_dim == 128

    cbm = build_cbm_pfi(config, device=torch.device("cpu"), logger=None)
    x = torch.randn(2, 3, 32, 32)
    x3 = torch.randn(2, 8, 8, 8)
    p3 = torch.randn(2, 4, 8, 8)
    m3 = torch.randn(2, 1, 8, 8)
    p3_out, aux = cbm.apply_p3_hook(x=x, x3=x3, p3=p3, m3=m3, training=True)

    assert torch.equal(p3_out, p3)
    assert aux["cbm_used"] is False
    assert cbm.compute_losses(aux, torch.zeros(2, 1, 32, 32)).shape == ()

    config.cbm_pfi_enable = True
    cbm = build_cbm_pfi(config, device=torch.device("cpu"), logger=None)
    cbm.memory.append_batch(
        x3=torch.randn(2, 8, 8, 8),
        p3=torch.randn(2, 4, 8, 8),
        gt=_square_gt(batch_size=2, height=32, width=32),
        img_ids=["img-a", "img-b"],
    )
    cbm.memory.finalize(device=torch.device("cpu"), dtype=torch.float32)
    cbm.prepare_epoch(model=None, labeled_loader=None, epoch=5)

    p3_corr, aux = cbm.apply_p3_hook(x=x, x3=x3, p3=p3, m3=m3, training=True)
    assert aux["cbm_used"] is True
    assert p3_corr.shape == p3.shape
    assert aux["num_memory_tokens"] > 0
    p1_out = torch.randn(2, 1, 32, 32)
    z_final = cbm.apply_final_fusion(p1_out, aux)
    assert z_final.shape == p1_out.shape
    assert aux["p_final"].shape == p1_out.shape


def test_build_pred_boundary_backward_from_sigmoid_source():
    source = torch.randn(2, 1, 16, 16, requires_grad=True)
    prob = torch.sigmoid(source)

    b_query, boundary_mask = build_pred_boundary(prob)

    assert boundary_mask.dtype == torch.bool
    loss = b_query.mean()
    loss.backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_cbm_runtime_backward_smoke_without_inplace_autograd_errors():
    class Config:
        cbm_pfi_enable = True
        cbm_print_diagnostics = False

    config = apply_cbm_defaults(Config())
    cbm = build_cbm_pfi(config, device=torch.device("cpu"), logger=None)
    cbm.memory.append_batch(
        x3=torch.randn(2, 8, 8, 8),
        p3=torch.randn(2, 4, 8, 8),
        gt=_square_gt(batch_size=2, height=32, width=32),
        img_ids=["img-a", "img-b"],
    )
    cbm.memory.finalize(device=torch.device("cpu"), dtype=torch.float32)
    cbm.prepare_epoch(model=None, labeled_loader=None, epoch=5)

    x = torch.randn(2, 3, 32, 32)
    x3 = torch.randn(2, 8, 8, 8)
    p3 = torch.randn(2, 4, 8, 8, requires_grad=True)
    m3 = torch.randn(2, 1, 8, 8, requires_grad=True)
    p1_out = torch.randn(2, 1, 32, 32, requires_grad=True)

    p3_corr, aux = cbm.apply_p3_hook(x=x, x3=x3, p3=p3, m3=m3, training=True)
    assert aux["cbm_used"] is True

    z_final = cbm.apply_final_fusion(p1_out, aux)
    loss_cbm = cbm.compute_losses(aux, _square_gt(batch_size=2, height=32, width=32))
    total = p3_corr.mean() + z_final.mean() + loss_cbm
    total.backward()

    assert p3.grad is not None
    assert m3.grad is not None
    assert p1_out.grad is not None
    assert torch.isfinite(p3.grad).all()
    assert torch.isfinite(m3.grad).all()
    assert torch.isfinite(p1_out.grad).all()
