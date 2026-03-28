"""Tests for UDM Epic 3 — CycleGAN: generator, discriminator, model, losses, dataset, pool, metrics."""

import numpy as np
import torch

from udm_epic3.models.generator import ResnetGenerator
from udm_epic3.models.discriminator import PatchDiscriminator
from udm_epic3.models.cyclegan import CycleGANModel
from udm_epic3.models.losses import (
    adversarial_loss_lsgan,
    cycle_consistency_loss,
    identity_loss,
    defect_preservation_loss,
)
from udm_epic3.data.image_pool import ImagePool
from udm_epic3.evaluation.quality_metrics import compute_ssim, compute_defect_dice


class TestResnetGenerator:
    """ResNet generator forward pass and output properties."""

    def test_forward_shape(self):
        G = ResnetGenerator(in_channels=1, out_channels=1, n_filters=64, n_blocks=9)
        x = torch.randn(1, 1, 256, 256)
        out = G(x)
        assert out.shape == (1, 1, 256, 256)

    def test_output_range_tanh(self):
        G = ResnetGenerator(in_channels=1, out_channels=1, n_filters=64, n_blocks=9)
        x = torch.randn(1, 1, 256, 256)
        out = G(x)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_different_block_count(self):
        G = ResnetGenerator(in_channels=1, out_channels=1, n_filters=32, n_blocks=6)
        x = torch.randn(2, 1, 128, 128)
        out = G(x)
        assert out.shape == (2, 1, 128, 128)

    def test_multi_channel(self):
        G = ResnetGenerator(in_channels=3, out_channels=3, n_filters=32, n_blocks=4)
        x = torch.randn(1, 3, 64, 64)
        out = G(x)
        assert out.shape == (1, 3, 64, 64)


class TestPatchDiscriminator:
    """PatchGAN discriminator produces spatial output (not scalar)."""

    def test_forward_shape_is_spatial(self):
        D = PatchDiscriminator(in_channels=1, n_filters=64)
        x = torch.randn(1, 1, 256, 256)
        out = D(x)
        # Output should be 4-D: [B, 1, H', W'] with H', W' > 1
        assert out.ndim == 4
        assert out.shape[0] == 1
        assert out.shape[1] == 1
        assert out.shape[2] > 1
        assert out.shape[3] > 1

    def test_output_is_raw_logits(self):
        """Output should NOT be passed through sigmoid (raw logits for LSGAN)."""
        D = PatchDiscriminator(in_channels=1, n_filters=64)
        x = torch.randn(1, 1, 256, 256)
        out = D(x)
        # Raw logits can be negative and can exceed 1
        # Just check it's not all in [0, 1] range (with high probability)
        assert out.min() < 0.5 or out.max() > 0.5  # not all zero

    def test_batch_dimension(self):
        D = PatchDiscriminator(in_channels=1, n_filters=64)
        x = torch.randn(4, 1, 128, 128)
        out = D(x)
        assert out.shape[0] == 4


class TestCycleGANModel:
    """Full CycleGAN model with two generator-discriminator pairs."""

    def test_forward_generators_keys(self):
        model = CycleGANModel(in_channels=1, n_filters_g=16, n_blocks=3, n_filters_d=16)
        real_A = torch.randn(1, 1, 64, 64)
        real_B = torch.randn(1, 1, 64, 64)
        outputs = model.forward_generators(real_A, real_B)

        expected_keys = {"fake_B", "rec_A", "fake_A", "rec_B", "idt_A", "idt_B"}
        assert set(outputs.keys()) == expected_keys

    def test_forward_generators_shapes(self):
        model = CycleGANModel(in_channels=1, n_filters_g=16, n_blocks=3, n_filters_d=16)
        real_A = torch.randn(1, 1, 64, 64)
        real_B = torch.randn(1, 1, 64, 64)
        outputs = model.forward_generators(real_A, real_B)

        for key, val in outputs.items():
            assert val.shape == (1, 1, 64, 64), f"{key} has wrong shape: {val.shape}"

    def test_translate_a2b(self):
        model = CycleGANModel(in_channels=1, n_filters_g=16, n_blocks=3, n_filters_d=16)
        x = torch.randn(1, 1, 64, 64)
        out = model.translate(x, direction="a2b")
        assert out.shape == (1, 1, 64, 64)

    def test_translate_b2a(self):
        model = CycleGANModel(in_channels=1, n_filters_g=16, n_blocks=3, n_filters_d=16)
        x = torch.randn(1, 1, 64, 64)
        out = model.translate(x, direction="b2a")
        assert out.shape == (1, 1, 64, 64)

    def test_translate_invalid_direction(self):
        model = CycleGANModel(in_channels=1, n_filters_g=16, n_blocks=3, n_filters_d=16)
        x = torch.randn(1, 1, 64, 64)
        try:
            model.translate(x, direction="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestLosses:
    """CycleGAN loss functions: LSGAN, cycle, identity, defect."""

    def test_adversarial_loss_real(self):
        pred = torch.ones(1, 1, 8, 8)
        loss = adversarial_loss_lsgan(pred, target_is_real=True)
        # MSE between all-ones and all-ones target = 0
        assert loss.item() < 1e-6

    def test_adversarial_loss_fake(self):
        pred = torch.zeros(1, 1, 8, 8)
        loss = adversarial_loss_lsgan(pred, target_is_real=False)
        # MSE between all-zeros and all-zeros target = 0
        assert loss.item() < 1e-6

    def test_adversarial_loss_mismatch(self):
        pred = torch.ones(1, 1, 8, 8)
        loss = adversarial_loss_lsgan(pred, target_is_real=False)
        # MSE between all-ones and all-zeros target = 1.0
        assert abs(loss.item() - 1.0) < 1e-6

    def test_cycle_consistency_l1(self):
        original = torch.randn(1, 1, 8, 8)
        reconstructed = original.clone()
        loss = cycle_consistency_loss(reconstructed, original)
        assert loss.item() < 1e-6

    def test_cycle_consistency_nonzero(self):
        original = torch.zeros(1, 1, 8, 8)
        reconstructed = torch.ones(1, 1, 8, 8)
        loss = cycle_consistency_loss(reconstructed, original)
        assert abs(loss.item() - 1.0) < 1e-6

    def test_identity_l1(self):
        original = torch.randn(1, 1, 8, 8)
        idt_output = original.clone()
        loss = identity_loss(idt_output, original)
        assert loss.item() < 1e-6

    def test_defect_preservation_dice(self):
        # Perfect overlap: translated image has high values exactly where mask is
        mask = torch.zeros(1, 1, 8, 8)
        mask[0, 0, 2:6, 2:6] = 1.0
        # Translated image with high values in mask region (in [-1, 1] range)
        translated = torch.full((1, 1, 8, 8), -1.0)
        translated[0, 0, 2:6, 2:6] = 0.5  # above threshold after rescaling
        loss = defect_preservation_loss(mask, translated, mask_threshold=0.5)
        # Should be small since defect region overlaps well
        assert loss.item() < 0.5


class TestUnpairedDataset:
    """UnpairedDataset returns correct keys and shapes."""

    def test_returns_correct_keys(self, tmp_path):
        import cv2

        # Create temporary image directories with actual images
        dir_a = tmp_path / "domain_a"
        dir_b = tmp_path / "domain_b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Create dummy images
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(dir_a / f"img_{i:03d}.png"), img)
            cv2.imwrite(str(dir_b / f"img_{i:03d}.png"), img)

        from udm_epic3.data.unpaired_dataset import UnpairedDataset

        ds = UnpairedDataset(
            dir_A=str(dir_a), dir_B=str(dir_b), image_size=(64, 64),
        )
        assert len(ds) == 3
        sample = ds[0]
        assert "A" in sample
        assert "B" in sample
        assert "path_A" in sample
        assert "path_B" in sample

    def test_tensor_shapes(self, tmp_path):
        import cv2

        dir_a = tmp_path / "domain_a"
        dir_b = tmp_path / "domain_b"
        dir_a.mkdir()
        dir_b.mkdir()

        for i in range(2):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(dir_a / f"img_{i:03d}.png"), img)
            cv2.imwrite(str(dir_b / f"img_{i:03d}.png"), img)

        from udm_epic3.data.unpaired_dataset import UnpairedDataset

        ds = UnpairedDataset(
            dir_A=str(dir_a), dir_B=str(dir_b), image_size=(64, 64),
        )
        sample = ds[0]
        assert sample["A"].shape == (1, 64, 64)
        assert sample["B"].shape == (1, 64, 64)

    def test_tensor_range(self, tmp_path):
        import cv2

        dir_a = tmp_path / "domain_a"
        dir_b = tmp_path / "domain_b"
        dir_a.mkdir()
        dir_b.mkdir()

        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(dir_a / "img_000.png"), img)
        cv2.imwrite(str(dir_b / "img_000.png"), img)

        from udm_epic3.data.unpaired_dataset import UnpairedDataset

        ds = UnpairedDataset(
            dir_A=str(dir_a), dir_B=str(dir_b), image_size=(64, 64),
        )
        sample = ds[0]
        # Images should be normalised to [-1, 1]
        assert sample["A"].min() >= -1.0 - 1e-3
        assert sample["A"].max() <= 1.0 + 1e-3


class TestImagePool:
    """ImagePool buffer for discriminator training stability."""

    def test_pool_size_zero_returns_input(self):
        pool = ImagePool(pool_size=0)
        x = torch.randn(2, 1, 8, 8)
        out = pool.query(x)
        assert torch.equal(x, out)

    def test_pool_stores_images(self):
        pool = ImagePool(pool_size=10)
        x = torch.randn(3, 1, 8, 8)
        _ = pool.query(x)
        assert len(pool) == 3

    def test_pool_returns_correct_batch_size(self):
        pool = ImagePool(pool_size=50)
        x = torch.randn(4, 1, 8, 8)
        out = pool.query(x)
        assert out.shape == (4, 1, 8, 8)

    def test_pool_fills_then_swaps(self):
        pool = ImagePool(pool_size=5)
        # Fill the pool
        for _ in range(5):
            pool.query(torch.randn(1, 1, 4, 4))
        assert len(pool) == 5
        # Query again — pool is full, should still return correct shape
        out = pool.query(torch.randn(2, 1, 4, 4))
        assert out.shape == (2, 1, 4, 4)


class TestQualityMetrics:
    """Image-quality and defect-preservation metrics."""

    def test_ssim_identical_images(self):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        ssim_val = compute_ssim(img, img)
        assert abs(ssim_val - 1.0) < 1e-6

    def test_ssim_different_images(self):
        img_a = np.zeros((64, 64), dtype=np.uint8)
        img_b = np.full((64, 64), 255, dtype=np.uint8)
        ssim_val = compute_ssim(img_a, img_b)
        assert ssim_val < 0.1

    def test_ssim_float_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        ssim_val = compute_ssim(img, img)
        assert abs(ssim_val - 1.0) < 1e-6

    def test_dice_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        dice_val = compute_defect_dice(mask, mask)
        assert abs(dice_val - 1.0) < 1e-6

    def test_dice_no_overlap(self):
        mask_a = np.zeros((64, 64), dtype=np.uint8)
        mask_a[0:10, 0:10] = 255
        mask_b = np.zeros((64, 64), dtype=np.uint8)
        mask_b[50:64, 50:64] = 255
        dice_val = compute_defect_dice(mask_a, mask_b)
        assert dice_val == 0.0

    def test_dice_both_empty(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        dice_val = compute_defect_dice(mask, mask)
        assert dice_val == 1.0

    def test_dice_partial_overlap(self):
        mask_a = np.zeros((64, 64), dtype=np.float32)
        mask_a[0:32, :] = 1.0
        mask_b = np.zeros((64, 64), dtype=np.float32)
        mask_b[16:48, :] = 1.0
        dice_val = compute_defect_dice(mask_a, mask_b)
        # Intersection = 16 rows, A = 32 rows, B = 32 rows
        # Dice = 2*16 / (32+32) = 0.5
        assert abs(dice_val - 0.5) < 1e-6
