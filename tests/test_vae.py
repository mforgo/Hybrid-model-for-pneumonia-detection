"""Tests for ``pipeline.vae``: supervised VAE loss differentiability.

RED phase: ``pipeline.vae`` is an empty stub and torch is not installed in
the development environment, so this test fails at runtime until Task 7
implements the module. Collection must still succeed, hence all heavy
imports are lazy (function-level).
"""


def test_supervised_vae_loss_backward():
    """supervised_vae_loss must be differentiable end-to-end: loss.backward()
    must succeed and every trainable parameter must receive a finite gradient."""
    import torch

    from pipeline.vae import SupervisedVAE, supervised_vae_loss

    model = SupervisedVAE(input_dim=768, latent_dim=64)
    x = torch.randn(4, 768)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    loss = supervised_vae_loss(model, x, y, beta=0.001, lambda_clf=0.01)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)