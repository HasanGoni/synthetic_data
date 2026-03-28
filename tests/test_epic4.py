"""Tests for UDM Epic 4 — DANN: GRL, encoder, decoder, model, dataset, sampler, lambda, metrics."""

import math

import numpy as np
import torch
import torch.nn as nn

from udm_epic4.models.encoder import SharedEncoder
from udm_epic4.models.decoder import UNetDecoder
from udm_epic4.models.domain_classifier import (
    GradientReversalFunction,
    GradientReversalLayer,
    DomainClassifier,
)
from udm_epic4.models.dann import DANNModel
from udm_epic4.data.domain_sampler import DomainBatchSampler
from udm_epic4.training.lambda_scheduler import dann_lambda_schedule
from udm_epic4.evaluation.metrics import compute_f1, compute_iou, compute_dice


class TestGRL:
    """Gradient Reversal Layer correctness."""

    def test_forward_is_identity(self):
        x = torch.randn(2, 4, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        assert torch.allclose(x, y)

    def test_backward_flips_gradient(self):
        x = torch.randn(2, 4, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        loss = y.sum()
        loss.backward()
        # Gradient of sum w.r.t. x is all 1s; after reversal it should be all -1s
        assert x.grad is not None
        assert torch.allclose(x.grad, -torch.ones_like(x.grad))

    def test_lambda_scaling(self):
        x = torch.randn(2, 4, requires_grad=True)
        lam = 0.5
        y = GradientReversalFunction.apply(x, lam)
        loss = y.sum()
        loss.backward()
        expected = -lam * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_grl_module(self):
        grl = GradientReversalLayer(lambda_val=0.3)
        x = torch.randn(2, 4, requires_grad=True)
        y = grl(x)
        assert y.shape == x.shape


class TestEncoder:
    """Shared encoder produces correct output shapes."""

    def test_output_shapes(self):
        enc = SharedEncoder(backbone_name="convnext_tiny", pretrained=False, in_chans=3)
        x = torch.randn(1, 3, 128, 128)
        features = enc(x)
        assert len(features) == 4
        # Each feature should have batch dim 1
        for f in features:
            assert f.shape[0] == 1

    def test_feature_channels(self):
        enc = SharedEncoder(backbone_name="convnext_tiny", pretrained=False)
        channels = enc.feature_channels
        assert len(channels) == 4
        assert all(isinstance(c, int) and c > 0 for c in channels)

    def test_resnet_backbone(self):
        enc = SharedEncoder(backbone_name="resnet18", pretrained=False)
        x = torch.randn(1, 3, 128, 128)
        features = enc(x)
        assert len(features) == 4


class TestDecoder:
    """U-Net decoder skip connections and output shape."""

    def test_output_shape(self):
        enc = SharedEncoder(backbone_name="convnext_tiny", pretrained=False)
        dec = UNetDecoder(
            encoder_channels=enc.feature_channels,
            decoder_channels=[256, 128, 64, 32],
        )
        x = torch.randn(1, 3, 128, 128)
        features = enc(x)
        out = dec(features)
        assert out.shape[0] == 1
        assert out.shape[1] == 1  # single-channel segmentation


class TestDANN:
    """Full DANN model end-to-end."""

    def test_forward_shapes(self):
        model = DANNModel(
            backbone="convnext_tiny",
            pretrained=False,
            decoder_channels=[64, 32, 16, 8],
            domain_head_hidden=32,
        )
        x = torch.randn(2, 3, 128, 128)
        seg_logits, domain_logits = model(x, lambda_val=0.5)

        assert seg_logits.shape == (2, 1, 128, 128)
        assert domain_logits.shape == (2, 1)

    def test_encode(self):
        model = DANNModel(backbone="convnext_tiny", pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        bottleneck = model.encode(x)
        assert bottleneck.ndim == 4
        assert bottleneck.shape[0] == 1


class TestDomainSampler:
    """Balanced 50/50 source/target batch sampling."""

    def test_batch_composition(self):
        sampler = DomainBatchSampler(
            source_dataset_size=100,
            target_dataset_size=80,
            batch_size=8,
            drop_last=True,
        )
        batches = list(sampler)
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) == 8
            # First 4 should be source (0-99), last 4 should be target (100-179)
            source_indices = batch[:4]
            target_indices = batch[4:]
            assert all(0 <= i < 100 for i in source_indices)
            assert all(100 <= i < 180 for i in target_indices)

    def test_len(self):
        sampler = DomainBatchSampler(
            source_dataset_size=100,
            target_dataset_size=80,
            batch_size=8,
            drop_last=True,
        )
        assert len(sampler) > 0


class TestLambdaScheduler:
    """DANN lambda schedule correctness."""

    def test_start_near_zero(self):
        val = dann_lambda_schedule(0.0)
        assert abs(val) < 0.01

    def test_midpoint(self):
        val = dann_lambda_schedule(0.5)
        # At progress=0.5: 2/(1+exp(-5))-1 ≈ 0.986
        assert 0.5 < val < 1.0

    def test_end_near_max(self):
        val = dann_lambda_schedule(1.0)
        assert val > 0.99

    def test_custom_lambda_max(self):
        val = dann_lambda_schedule(1.0, lambda_max=0.5)
        assert 0.49 < val < 0.51

    def test_clamped(self):
        # Negative progress should clamp to 0
        val = dann_lambda_schedule(-0.5)
        assert abs(val) < 0.01
        # Progress > 1 should clamp to 1
        val = dann_lambda_schedule(2.0)
        assert val > 0.99


class TestMetrics:
    """Segmentation metric computation."""

    def test_perfect_f1(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        assert compute_f1(pred, target) == 1.0

    def test_zero_f1(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)
        assert compute_f1(pred, target) == 0.0

    def test_perfect_iou(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        assert compute_iou(pred, target) == 1.0

    def test_partial_overlap(self):
        pred = torch.zeros(1, 1, 8, 8)
        pred[0, 0, :4, :] = 1.0
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 2:6, :] = 1.0
        iou = compute_iou(pred, target)
        # Intersection = 2 rows, union = 6 rows => IoU = 2/6 = 0.333
        assert 0.3 < iou < 0.4

    def test_dice_equals_f1(self):
        pred = torch.zeros(1, 1, 8, 8)
        pred[0, 0, :4, :] = 1.0
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 2:6, :] = 1.0
        assert abs(compute_f1(pred, target) - compute_dice(pred, target)) < 1e-6

    def test_empty_both(self):
        pred = torch.zeros(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)
        assert compute_f1(pred, target) == 1.0
        assert compute_iou(pred, target) == 1.0

    def test_logits_input(self):
        # Negative logits -> sigmoid -> below threshold
        pred_logits = torch.full((1, 1, 8, 8), -5.0)
        target = torch.zeros(1, 1, 8, 8)
        assert compute_f1(pred_logits, target) == 1.0
