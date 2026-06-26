"""Declarative :class:`NodeSpec` for the v0.6.x node system.

A :class:`NodeSpec` is attached to every :class:`BaseNode` subclass
as the ``spec`` class attribute and is the single source of truth
for a node's identity, typed input/output contract, and tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["NodeSpec"]


@dataclass
class NodeSpec:
    """Declarative description of a node.

    Attributes:
        type: Stable, unique node type identifier, e.g.
            ``"image_txt2img"`` or ``"text_chat"``.  Used as the
            :class:`ModuleBus` name under the ``"node"`` kind.
        name: Human-readable display name.
        description: One-line description of what the node does.
        inputs: Mapping of input name to its declared port type string
            (e.g. ``"TEXT"``, ``"IMAGE"``, ``"Optional[SEED]"``).
            Optional inputs are expressed with the ``Optional[T]``
            wrapper.
        outputs: Mapping of output name to its declared port type string.
        tags: Free-form tags used for discovery / filtering.
    """

    type: str
    name: str
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the spec fields after dataclass initialisation."""
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("NodeSpec.type must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("NodeSpec.name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise ValueError("NodeSpec.description must be a string.")
        if not isinstance(self.inputs, dict):
            raise ValueError("NodeSpec.inputs must be a dict[str, str].")
        if not isinstance(self.outputs, dict):
            raise ValueError("NodeSpec.outputs must be a dict[str, str].")
        if not isinstance(self.tags, list):
            raise ValueError("NodeSpec.tags must be a list[str].")

    def __repr__(self) -> str:
        return (
            "NodeSpec(type={!r}, name={!r}, "
            "inputs={}, outputs={}, tags={!r})".format(
                self.type,
                self.name,
                list(self.inputs.keys()),
                list(self.outputs.keys()),
                self.tags,
            )
        )
