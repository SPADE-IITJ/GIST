"""Region adapters (adaptation.region_adapter; phase 1)."""
from __future__ import annotations

import torch

from gist.adaptation.region_adapter import RegionAdapter
from gist.adaptation.subgraph import audit_exactness


def _assignment():
    a = torch.full((30,), -1, dtype=torch.long)
    a[:10], a[10:20] = 0, 1                              # nodes 20-29 belong to no community
    return a


def test_zero_initialised_adapter_is_a_no_op():
    a = _assignment()
    ad = RegionAdapter(4, lambda: a)
    logits = torch.randn(30, 4)
    assert torch.equal(ad.apply(logits, torch.arange(30)), logits)


def test_update_moves_only_the_communities_it_trained_on():
    torch.manual_seed(0)
    a = _assignment()
    ad = RegionAdapter(4, lambda: a, lr=0.05, steps=5)
    logits, ids = torch.randn(30, 4), torch.arange(30)
    ad.refit(logits[:10], torch.randint(0, 4, (10,)), ids[:10])          # community 0 only
    after0 = ad.apply(logits, ids)
    assert not torch.equal(after0[:10], logits[:10])
    assert torch.equal(after0[10:], logits[10:])                        # community 1 and unassigned untouched
    ad.refit(logits[10:20], torch.randint(0, 4, (10,)), ids[10:20])     # now community 1 only
    after1 = ad.apply(logits, ids)
    assert torch.equal(after1[:10], after0[:10])                        # no momentum or decay leaks into community 0
    assert not torch.equal(after1[10:20], logits[10:20])
    assert torch.equal(after1[20:], logits[20:])


def test_storage_grows_for_new_community_ids():
    a = torch.tensor([0, 200, 5], dtype=torch.long)
    ad = RegionAdapter(3, lambda: a, capacity=8)
    out = ad.apply(torch.randn(3, 3), torch.arange(3))
    assert out.shape == (3, 3) and ad.capacity > 200


def test_subgraph_update_stays_exact_with_an_adapter():
    from gist.models.registry import build_backbone
    torch.manual_seed(1)
    n, f, c = 60, 8, 4
    x, y = torch.randn(n, f), torch.randint(0, c, (n,))
    edge_index = torch.randint(0, n, (2, 240))
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    assign = torch.arange(n) % 5
    ad = RegionAdapter(c, lambda: assign)
    with torch.no_grad():
        ad.W.normal_(0, 0.3)
        ad.b.normal_(0, 0.3)
    for name in ("gat", "sage", "gcn", "gin"):
        model = build_backbone(name, f, c, 16, 2)
        r = audit_exactness(model, x, y, edge_index, n, torch.tensor([3, 17, 41]), 2, adapter=ad)
        assert r["rel_diff_global"] < 1e-8, (name, r)
        assert r["rel_diff_short_global"] > 1e-3, (name, r)


def test_runner_with_adapters_leaves_the_baseline_unchanged():
    from tests.test_oracle import _small_run
    cfg, ds, plan, warm, run_arm = _small_run()
    ref = run_arm("baseline", cfg, ds, plan, warm)
    cfg.adaptation.region_adapter = True
    base = run_arm("baseline", cfg, ds, plan, warm)
    assert [s.accuracy for s in base.steps] == [s.accuracy for s in ref.steps]
    for arm in ("full", "selective", "random_region", "iso_full"):
        r = run_arm(arm, cfg, ds, plan, warm)
        assert len(r.steps) == plan.steps
