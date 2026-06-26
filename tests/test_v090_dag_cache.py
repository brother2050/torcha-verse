"""Tests for v0.9.0 DAG + HierarchicalCache (20 tests)."""
from __future__ import annotations

import json
import threading

import pytest

from pipeline.cache import CacheStats, HierarchicalCache, compute_fingerprint
from pipeline.dag import DAG, DAGEdge, DAGNode


# ---------------------------------------------------------------------------
# Section 1 - compute_fingerprint
# ---------------------------------------------------------------------------
def test_fp_same_input_same_output():
    fp1 = compute_fingerprint("n1", args=(1, 2), kwargs={"a": 1}, parent_fingerprints=["p"])
    fp2 = compute_fingerprint("n1", args=(1, 2), kwargs={"a": 1}, parent_fingerprints=["p"])
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_fp_different_input_differs():
    a = compute_fingerprint("n", args=(1, 2), kwargs={}, parent_fingerprints=())
    b = compute_fingerprint("n", args=(1, 3), kwargs={}, parent_fingerprints=())
    assert a != b


def test_fp_different_node_id_differs():
    a = compute_fingerprint("n1", args=(1,), kwargs={}, parent_fingerprints=())
    b = compute_fingerprint("n2", args=(1,), kwargs={}, parent_fingerprints=())
    assert a != b


def test_fp_parent_propagation():
    a = compute_fingerprint("child", args=(1,), kwargs={}, parent_fingerprints=["p1"])
    b = compute_fingerprint("child", args=(1,), kwargs={}, parent_fingerprints=["p2"])
    assert a != b


def test_fp_stable_across_calls():
    base = compute_fingerprint("n", args=(1, 2), kwargs={"k": 1}, parent_fingerprints=["x"])
    for _ in range(1000):
        cur = compute_fingerprint("n", args=(1, 2), kwargs={"k": 1}, parent_fingerprints=["x"])
        assert cur == base


def test_fp_handles_complex_kwargs():
    payload = {
        "d": {"a": 1, "b": [1, 2, 3]},
        "l": [1, "x", {"k": (1, 2)}],
        "s": {1, 2, 3},
        "b": b"hello-bytes-blob",
        "nested": {"inner": {"deep": [4, 5, {"k": "v"}]}},
    }
    fp1 = compute_fingerprint("n", args=(), kwargs=payload, parent_fingerprints=())
    fp2 = compute_fingerprint("n", args=(), kwargs=payload, parent_fingerprints=())
    assert fp1 == fp2
    # Round-trip through json to ensure payload itself is dump-able.
    json.dumps(payload, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Section 2 - HierarchicalCache behavior
# ---------------------------------------------------------------------------
def test_cache_miss_returns_none():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    assert cache.get_output("nonexistent") is None


def test_cache_hit_returns_value():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    cache.put_output("fp1", {"v": 42})
    assert cache.get_output("fp1") == {"v": 42}


def test_cache_lru_eviction():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=4)
    for i in range(4):
        cache.put_output(f"fp{i}", i)
    assert cache.get_output("fp0") == 0  # recent touch
    cache.put_output("fp4", 4)  # 5th item; capacity is 4 -> evict LRU
    # fp0 was touched last, so it should still be there; fp1 should be evicted
    assert cache.get_output("fp0") == 0
    assert cache.get_output("fp1") is None
    assert cache.get_output("fp4") == 4


def test_cache_invalidate_all():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    cache.put_output("a", 1)
    cache.put_object("m", "weights")
    cache.invalidate_all()
    assert cache.get_output("a") is None
    assert cache.get_object("m") is None


def test_cache_stats_hit_miss():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    cache.put_output("a", 1)
    cache.get_output("a")  # hit
    cache.get_output("a")  # hit
    cache.get_output("zz")  # miss (output)
    cache.put_object("m", "x")
    cache.get_object("m")  # hit (object)
    cache.get_object("absent")  # miss (object)
    snap = cache.stats_snapshot()
    assert snap.hits == 3
    assert snap.misses == 2
    assert isinstance(snap, CacheStats)


def test_cache_objects_bucket():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    cache.put_object("vae", "vae-weights")
    cache.put_object("clip", "clip-weights")
    assert cache.get_object("vae") == "vae-weights"
    assert cache.get_object("clip") == "clip-weights"
    # putting output with the same key string must not collide with the object bucket
    cache.put_output("vae", {"latents": 1})
    assert cache.get_output("vae") == {"latents": 1}
    assert cache.get_object("vae") == "vae-weights"


def test_cache_object_invalidate():
    cache = HierarchicalCache(capacity_outputs=4, capacity_objects=2)
    cache.put_object("vae", "weights")
    assert cache.get_object("vae") == "weights"
    cache.invalidate_object("vae")
    assert cache.get_object("vae") is None


def test_cache_concurrent_safe():
    cache = HierarchicalCache(capacity_outputs=128, capacity_objects=16)
    errors = []

    def worker(idx: int) -> None:
        try:
            for i in range(100):
                key = f"k-{idx}-{i % 17}"
                cache.put_output(key, idx * 1000 + i)
                _ = cache.get_output(key)
        except Exception as exc:  # pragma: no cover - smoke only
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "thread deadlocked"
    assert not errors, f"concurrent workers raised: {errors!r}"


# ---------------------------------------------------------------------------
# Section 3 - DAG integration
# ---------------------------------------------------------------------------
def test_dag_node_imports():
    node = DAGNode(id="a", node_type="text_chat")
    edge = DAGEdge(from_node="a", to_node="b", output_key="text", input_key="prompt")
    dag = DAG()
    assert node.id == "a"
    assert edge.from_node == "a"
    assert dag.node_count == 0


def test_dag_topological_sort():
    dag = DAG()
    ids = ["n0", "n1", "n2", "n3", "n4"]
    for nid in ids:
        dag.add_node(DAGNode(id=nid, node_type="text_chat"))
    # n1 depends on n0, n2 on n1, n3 on n2, n4 on n3
    dag.add_node(DAGNode(id="n1", node_type="text_chat", dependencies=["n0"]))
    dag.add_node(DAGNode(id="n2", node_type="text_chat", dependencies=["n1"]))
    dag.add_node(DAGNode(id="n3", node_type="text_chat", dependencies=["n2"]))
    dag.add_node(DAGNode(id="n4", node_type="text_chat", dependencies=["n3"]))
    order = dag.topological_sort()
    for i in range(len(order) - 1):
        assert order.index(f"n{i}") < order.index(f"n{i + 1}")


def test_dag_cycle_detect():
    dag = DAG()
    dag.add_node(DAGNode(id="a", node_type="text_chat", dependencies=["c"]))
    dag.add_node(DAGNode(id="b", node_type="text_chat", dependencies=["a"]))
    dag.add_node(DAGNode(id="c", node_type="text_chat", dependencies=["b"]))
    with pytest.raises(ValueError):
        dag.topological_sort()


def test_dag_parallel_groups():
    dag = DAG()
    dag.add_node(DAGNode(id="root", node_type="text_chat"))
    # Fan out: three children of root -> all should be in the same parallel group.
    for child in ("c1", "c2", "c3"):
        dag.add_node(DAGNode(id=child, node_type="text_chat", dependencies=["root"]))
    groups = dag.parallel_groups()
    # Find the layer containing all three children.
    flat = [nid for g in groups for nid in g]
    assert {"c1", "c2", "c3"}.issubset(set(flat))
    # All three must share a group (same depth).
    depth_of = {nid: idx for idx, g in enumerate(groups) for nid in g}
    assert depth_of["c1"] == depth_of["c2"] == depth_of["c3"]


def test_dag_validate_missing_dep():
    dag = DAG()
    dag.add_node(DAGNode(id="a", node_type="text_chat", dependencies=["ghost"]))
    errors = dag.validate()
    assert any("ghost" in e for e in errors)


def test_dag_mermaid_export():
    dag = DAG()
    dag.add_node(DAGNode(id="a", node_type="text_chat"))
    dag.add_node(DAGNode(id="b", node_type="text_chat", dependencies=["a"]))
    out = dag.visualize()
    assert isinstance(out, str)
    assert out != ""
    assert "graph TD" in out
