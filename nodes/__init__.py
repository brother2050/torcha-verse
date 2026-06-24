"""L4 Capability-layer node system for TorchaVerse v0.3.0.

This package replaces the v0.1.0 "god-class" engines
(``text_engine.py`` / ``image_engine.py`` / ``audio_engine.py`` /
``video_engine.py`` ...) with a set of small, single-responsibility,
composable *nodes*.  Each node declares a typed input/output contract
through a :class:`NodeSpec`, executes one well-defined operation against
a :class:`NodeContext`, and is discovered through the
:class:`NodeRegistry` (backed by :class:`ModuleBus`).

Layering (L1 -> L4):

* L1 ``infrastructure`` -- config, logging, devices, resource budgets.
* L2 ``assets`` -- the asset model + :class:`AssetStore`.
* L3 ``core`` -- :class:`ModuleBus`, model registry, schedulers.
* L4 ``nodes`` (this package) -- composable capability nodes.

Importing this package eagerly imports every node submodule so that the
``@register_node`` decorators run and every node is registered with the
process-wide :class:`ModuleBus`.  As a result a freshly constructed
:class:`NodeRegistry` immediately sees the full node catalogue::

    from nodes import NodeRegistry
    registry = NodeRegistry()
    print(len(registry.list()))   # all registered nodes

Node catalogue
--------------
* **Text** -- :class:`TextNode`, :class:`TextCompletionNode`
* **Image** -- :class:`ImageTxt2ImgNode`, :class:`ImageImg2ImgNode`,
  :class:`ImageUpscaleNode`, :class:`ImageInpaintNode`
* **Video** -- :class:`VideoTxt2VidNode`, :class:`VideoInterpolateNode`,
  :class:`VideoStitchNode`
* **Audio** -- :class:`AudioTTSNode`, :class:`AudioMusicNode`
* **Subtitle** -- :class:`SubtitleGenerateNode`,
  :class:`SubtitleTranslateNode`, :class:`SubtitleBurnNode`,
  :class:`SubtitleExportNode`
* **Consistency** -- :class:`CharacterApplyNode`,
  :class:`OutfitApplyNode`, :class:`SceneApplyNode`,
  :class:`DepthConditionNode`, :class:`FiveViewNode`
* **Digital Human** -- :class:`LipSyncNode`, :class:`TalkingHeadNode`,
  :class:`PortraitAnimateNode`, :class:`DigitalHumanNode`,
  :class:`FaceEnhanceNode`, :class:`VoiceCloneNode`
* **Export** -- :class:`ExportImageNode`, :class:`ExportVideoNode`,
  :class:`ExportAudioNode`
"""

from __future__ import annotations

# Base infrastructure (must be imported first so the submodules can use
# ``register_node`` / ``BaseNode`` / ``NodeSpec`` / ``NodeContext``).
from .base import (
    BaseNode,
    NodeContext,
    NodeRegistry,
    NodeSpec,
    register_node,
)

# Eagerly import every node submodule so that the @register_node
# decorators execute and the nodes appear on the ModuleBus.
from . import audio as audio  # noqa: F401
from . import consistency as consistency  # noqa: F401
from . import digital_human as digital_human  # noqa: F401
from . import export as export  # noqa: F401
from . import image as image  # noqa: F401
from . import subtitle as subtitle  # noqa: F401
from . import text as text  # noqa: F401
from . import video as video  # noqa: F401

# Re-export the concrete node classes for convenient ``from nodes import X``.
from .audio import AudioMusicNode, AudioTTSNode
from .consistency import (
    CharacterApplyNode,
    DepthConditionNode,
    FiveViewNode,
    OutfitApplyNode,
    SceneApplyNode,
)
from .digital_human import (
    DigitalHumanNode,
    FaceEnhanceNode,
    LipSyncNode,
    PortraitAnimateNode,
    TalkingHeadNode,
    VoiceCloneNode,
)
from .export import ExportAudioNode, ExportImageNode, ExportVideoNode
from .image import (
    ImageImg2ImgNode,
    ImageInpaintNode,
    ImageTxt2ImgNode,
    ImageUpscaleNode,
)
from .subtitle import (
    SubtitleBurnNode,
    SubtitleExportNode,
    SubtitleGenerateNode,
    SubtitleTranslateNode,
)
from .text import TextCompletionNode, TextNode
from .video import VideoInterpolateNode, VideoStitchNode, VideoTxt2VidNode

__all__ = [
    # Base infrastructure
    "NodeSpec",
    "NodeContext",
    "BaseNode",
    "NodeRegistry",
    "register_node",
    # Text nodes
    "TextNode",
    "TextCompletionNode",
    # Image nodes
    "ImageTxt2ImgNode",
    "ImageImg2ImgNode",
    "ImageUpscaleNode",
    "ImageInpaintNode",
    # Video nodes
    "VideoTxt2VidNode",
    "VideoInterpolateNode",
    "VideoStitchNode",
    # Audio nodes
    "AudioTTSNode",
    "AudioMusicNode",
    # Subtitle nodes
    "SubtitleGenerateNode",
    "SubtitleTranslateNode",
    "SubtitleBurnNode",
    "SubtitleExportNode",
    # Consistency nodes
    "CharacterApplyNode",
    "OutfitApplyNode",
    "SceneApplyNode",
    "DepthConditionNode",
    "FiveViewNode",
    # Digital human nodes
    "LipSyncNode",
    "TalkingHeadNode",
    "PortraitAnimateNode",
    "DigitalHumanNode",
    "FaceEnhanceNode",
    "VoiceCloneNode",
    # Export nodes
    "ExportImageNode",
    "ExportVideoNode",
    "ExportAudioNode",
]
