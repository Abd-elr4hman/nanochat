"""
Smoke tests for the Mamba-2 wrapper. Example run:

    python -m pytest tests/test_mamba_smoke.py -v

Skipped automatically if mambapy/mamba_ssm cannot be imported (e.g. on Windows
without WSL or before runs/setup_mamba.sh has been run).
"""

import pytest
import torch

mamba = pytest.importorskip(
    "nanochat.mamba",
    reason="Mamba-2 backbone requires mamba_ssm + mambapy (Linux/WSL only)",
)
Mamba = mamba.Mamba
MambaConfig = mamba.MambaConfig


def test_meta_build():
    """Wrapper constructs on the meta device without materializing tensors."""
    cfg = MambaConfig(sequence_len=128, vocab_size=1024, n_layer=2, n_embd=256)
    with torch.device("meta"):
        m = Mamba(cfg)
    # Sanity: at least the input/output projection matrices are 2D.
    shapes = {tuple(p.shape) for p in m.parameters()}
    assert any(len(s) == 2 for s in shapes), "expected at least one 2D matrix"
    assert (640, 1, 4) in shapes, "expected the 3D conv1d weight at (d_inner+2*n_groups*d_state, 1, d_conv)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_forward_backward_cuda():
    """Forward + backward run on GPU through the Triton kernel."""
    cfg = MambaConfig(sequence_len=128, vocab_size=1024, n_layer=2, n_embd=256)
    m = Mamba(cfg).cuda()
    m.init_weights()
    idx = torch.randint(0, 1024, (2, 128), device="cuda")
    targets = torch.randint(0, 1024, (2, 128), device="cuda")
    loss = m(idx, targets)
    # Random init over vocab=1024 should produce loss ≈ ln(1024) ≈ 6.93.
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    expected = torch.tensor(1024.0).log().item()
    assert abs(loss.item() - expected) < 0.5, f"untrained loss {loss.item()} far from ln(vocab)={expected}"
    loss.backward()
    # All trainable params should have grads.
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
