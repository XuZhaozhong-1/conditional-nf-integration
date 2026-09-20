import pytest
import torch

from nf import ConditionalCubicFlow1D, ConditionalLinearFlow1D


@pytest.mark.parametrize("flow_type", [ConditionalLinearFlow1D, ConditionalCubicFlow1D])
@pytest.mark.parametrize("column", [False, True])
def test_round_trip_and_logdet(flow_type, column):
    torch.manual_seed(3)
    flow = flow_type(cond_dim=2, hidden=16, n_bins=6).double()
    u = torch.linspace(0, 1, 101, dtype=torch.double)
    if column:
        u = u[:, None]
    cond = torch.tensor([[0.2, -0.7]], dtype=torch.double)
    x, forward_logdet = flow(u, cond)
    recovered, inverse_forward_logdet = flow.inverse(x, cond)
    assert torch.allclose(recovered, u, atol=2e-10, rtol=2e-10)
    assert torch.allclose(inverse_forward_logdet, forward_logdet, atol=2e-9, rtol=2e-9)
    assert torch.all(x.reshape(-1)[1:] > x.reshape(-1)[:-1])
    assert torch.allclose(flow.log_prob(x, cond), -forward_logdet, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("flow_type", [ConditionalLinearFlow1D, ConditionalCubicFlow1D])
def test_sample_and_gradients(flow_type):
    flow = flow_type(cond_dim=2, hidden=16, n_bins=4)
    cond = torch.randn(1, 2)
    samples = flow.sample(32, cond)
    assert samples.shape == (32, 1)
    assert torch.all((samples >= 0) & (samples <= 1))
    loss = -flow.log_prob(torch.linspace(0.05, 0.95, 20)[:, None], cond).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
