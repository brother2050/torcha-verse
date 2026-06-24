"""Tests for v0.3.0 template-node consistency.

Verifies that every built-in pipeline template uses only ``node_type``
identifiers that are actually registered in the :class:`NodeRegistry`,
and that every edge's ``output_key`` / ``input_key`` matches the
declared ``NodeSpec.inputs`` / ``NodeSpec.outputs`` of the connected
nodes.
"""
from __future__ import annotations

import pytest

# Importing the nodes package triggers every @register_node decorator.
import nodes  # noqa: F401
from nodes import NodeRegistry
from pipeline.templates import BUILTIN_TEMPLATES, TemplateRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def registry():
    """Return a NodeRegistry backed by the global ModuleBus."""
    return NodeRegistry()


@pytest.fixture()
def spec_map(registry):
    """Return a dict mapping node_type -> NodeSpec for all registered nodes."""
    return {spec.type: spec for spec in registry.list()}


@pytest.fixture()
def templates():
    """Return the list of all built-in PipelineTemplate instances."""
    return BUILTIN_TEMPLATES


# ---------------------------------------------------------------------------
# Node type registration
# ---------------------------------------------------------------------------
class TestTemplateNodeTypesRegistered:
    """Every node_type in every template must be a registered node."""

    def test_at_least_12_templates(self, templates):
        """The built-in catalogue ships with at least 12 templates."""
        assert len(templates) >= 12

    def test_all_node_types_registered(self, templates, spec_map):
        """Every node_type referenced by a template is in the NodeRegistry."""
        unregistered = []
        for tmpl in templates:
            for node in tmpl.dag_dict.get("nodes", []):
                nt = node["node_type"]
                if nt not in spec_map:
                    unregistered.append(
                        f"{tmpl.name}: node '{node['id']}' uses "
                        f"unregistered node_type '{nt}'"
                    )
        assert not unregistered, (
            "Unregistered node_types found:\n" + "\n".join(unregistered)
        )


# ---------------------------------------------------------------------------
# Edge port-key consistency
# ---------------------------------------------------------------------------
class TestTemplateEdgePortKeys:
    """Every edge port key must match the connected node's spec."""

    def test_output_keys_match_source_outputs(self, templates, spec_map):
        """Each edge output_key is declared in the from_node's spec.outputs."""
        mismatches = []
        for tmpl in templates:
            nodes_by_id = {
                n["id"]: n for n in tmpl.dag_dict.get("nodes", [])
            }
            for edge in tmpl.dag_dict.get("edges", []):
                from_id = edge["from_node"]
                out_key = edge["output_key"]
                node = nodes_by_id.get(from_id)
                if node is None:
                    continue
                spec = spec_map.get(node["node_type"])
                if spec is None:
                    continue
                if out_key not in spec.outputs:
                    mismatches.append(
                        f"{tmpl.name}: edge {from_id}->{edge['to_node']} "
                        f"output_key '{out_key}' not in "
                        f"{node['node_type']} outputs {list(spec.outputs)}"
                    )
        assert not mismatches, (
            "Invalid output_keys found:\n" + "\n".join(mismatches)
        )

    def test_input_keys_match_target_inputs(self, templates, spec_map):
        """Each edge input_key is declared in the to_node's spec.inputs."""
        mismatches = []
        for tmpl in templates:
            nodes_by_id = {
                n["id"]: n for n in tmpl.dag_dict.get("nodes", [])
            }
            for edge in tmpl.dag_dict.get("edges", []):
                to_id = edge["to_node"]
                in_key = edge["input_key"]
                node = nodes_by_id.get(to_id)
                if node is None:
                    continue
                spec = spec_map.get(node["node_type"])
                if spec is None:
                    continue
                if in_key not in spec.inputs:
                    mismatches.append(
                        f"{tmpl.name}: edge {edge['from_node']}->{to_id} "
                        f"input_key '{in_key}' not in "
                        f"{node['node_type']} inputs {list(spec.inputs)}"
                    )
        assert not mismatches, (
            "Invalid input_keys found:\n" + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# DAG structural validation
# ---------------------------------------------------------------------------
class TestTemplateDagValidation:
    """Each template's materialised DAG must pass structural validation."""

    def test_all_templates_validate(self, templates):
        """Every template converts to a DAG with zero validation errors."""
        errors_all = []
        for tmpl in templates:
            dag = tmpl.to_dag()
            errors = dag.validate()
            if errors:
                errors_all.append(
                    f"{tmpl.name}: {errors}"
                )
        assert not errors_all, (
            "DAG validation errors:\n" + "\n".join(errors_all)
        )

    def test_all_templates_topological_sort(self, templates):
        """Every template's DAG can be topologically sorted."""
        for tmpl in templates:
            dag = tmpl.to_dag()
            order = dag.topological_sort()
            assert len(order) == len(dag.node_ids)


# ---------------------------------------------------------------------------
# Per-template parametrised checks
# ---------------------------------------------------------------------------
_TEMPLATE_NAMES = [t.name for t in BUILTIN_TEMPLATES]


@pytest.mark.parametrize("tmpl_name", _TEMPLATE_NAMES)
def test_template_node_types_registered(tmpl_name, spec_map):
    """Parametrised: every node_type in a single template is registered."""
    reg = TemplateRegistry()
    tmpl = reg.get(tmpl_name)
    for node in tmpl.dag_dict.get("nodes", []):
        assert node["node_type"] in spec_map, (
            f"Template '{tmpl_name}': node '{node['id']}' uses "
            f"unregistered node_type '{node['node_type']}'"
        )


@pytest.mark.parametrize("tmpl_name", _TEMPLATE_NAMES)
def test_template_edge_keys_valid(tmpl_name, spec_map):
    """Parametrised: every edge port key in a single template is valid."""
    reg = TemplateRegistry()
    tmpl = reg.get(tmpl_name)
    nodes_by_id = {n["id"]: n for n in tmpl.dag_dict.get("nodes", [])}
    for edge in tmpl.dag_dict.get("edges", []):
        from_node = nodes_by_id.get(edge["from_node"])
        to_node = nodes_by_id.get(edge["to_node"])
        if from_node:
            from_spec = spec_map.get(from_node["node_type"])
            if from_spec:
                assert edge["output_key"] in from_spec.outputs, (
                    f"Template '{tmpl_name}': edge "
                    f"{edge['from_node']}->{edge['to_node']} "
                    f"output_key '{edge['output_key']}' not in "
                    f"{from_node['node_type']}.outputs "
                    f"{list(from_spec.outputs)}"
                )
        if to_node:
            to_spec = spec_map.get(to_node["node_type"])
            if to_spec:
                assert edge["input_key"] in to_spec.inputs, (
                    f"Template '{tmpl_name}': edge "
                    f"{edge['from_node']}->{edge['to_node']} "
                    f"input_key '{edge['input_key']}' not in "
                    f"{to_node['node_type']}.inputs "
                    f"{list(to_spec.inputs)}"
                )
