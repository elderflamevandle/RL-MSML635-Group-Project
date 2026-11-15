"""Unit tests for the HF/PEFT GRPO fallback trainer (src.training.grpo_hf).

These are deterministic -- unlike a real smoke test, they don't depend on an
untrained base model happening to sample a reward-earning response. They
mock out generation and drive the math/update machinery directly with
synthetic data, so they run on CPU in well under a second and don't need a
GPU, a HF download, or any particular amount of RAM.
"""
import math

from src.training.grpo_hf import group_advantages


def test_group_advantages_zero_variance_group_gives_zero_advantage():
    # Every rollout in the group scores identically -> no signal, by design.
    rewards = [0.5, 0.5, 0.5, 0.5]
    advs = group_advantages(rewards, n=2)
    assert advs == [0.0, 0.0, 0.0, 0.0]


def test_group_advantages_centers_and_scales_within_each_group():
    # Two groups of 2: group A has spread, group B is identical to A's mean.
    rewards = [0.0, 1.0, 0.5, 0.5]
    advs = group_advantages(rewards, n=2)
    # Group A: mean=0.5, std=0.5 -> advantages are -1, +1 (up to the eps floor).
    assert math.isclose(advs[0], -1.0, abs_tol=1e-4)
    assert math.isclose(advs[1], 1.0, abs_tol=1e-4)
    # Group B: zero variance -> zero advantage (eps floor only avoids 0/0).
    assert advs[2] == 0.0 and advs[3] == 0.0


def test_group_advantages_is_independent_across_groups():
    # A high-reward group and a low-reward group should normalize to the
    # same shape -- GRPO's whole point is comparing within a group, not
    # across groups (otherwise the easy group would dominate the gradient).
    rewards = [10.0, 20.0, -5.0, -3.0]
    advs = group_advantages(rewards, n=2)
    assert math.isclose(advs[0], -advs[1], abs_tol=1e-6)
    assert math.isclose(advs[2], -advs[3], abs_tol=1e-6)


def test_ppo_clip_backward_moves_params_toward_higher_advantage_token(tmp_path):
    """End-to-end on a toy 2-token-vocab 'model' (no HF download needed):
    confirms the clipped-surrogate loss actually produces gradients that
    push up the log-prob of the token that got the positive advantage, and
    pushes down the one with negative advantage -- the one thing the real
    trainer depends on that no amount of config-wiring can verify."""
    import torch

    torch.manual_seed(0)
    vocab, hidden = 2, 4
    embed = torch.nn.Embedding(vocab, hidden)
    head = torch.nn.Linear(hidden, vocab, bias=False)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=0.05)  # small step: just check direction, not magnitude

    def logprob_of_token(token: int) -> torch.Tensor:
        logits = head(embed(torch.tensor([0])))  # single fixed "state"
        return torch.log_softmax(logits, dim=-1)[0, token]

    # Two "rollouts" from the same group: token 0 got reward -1, token 1 got +1.
    advs = group_advantages([-1.0, 1.0], n=2)
    assert math.isclose(advs[0], -1.0, abs_tol=1e-4)
    assert math.isclose(advs[1], 1.0, abs_tol=1e-4)

    old_lp = [logprob_of_token(0).detach(), logprob_of_token(1).detach()]
    lp0_before, lp1_before = old_lp[0].item(), old_lp[1].item()

    opt.zero_grad()
    for token, adv, olp in zip([0, 1], advs, old_lp):
        new_lp = logprob_of_token(token)
        ratio = torch.exp(new_lp - olp)
        surr = torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
        (-surr).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
    assert float(grad_norm) > 0, "PPO-clip loss produced no gradient at all"
    opt.step()

    lp0_after, lp1_after = logprob_of_token(0).item(), logprob_of_token(1).item()
    assert lp1_after > lp1_before, "positive-advantage token should become more likely"
    assert lp0_after < lp0_before, "negative-advantage token should become less likely"
