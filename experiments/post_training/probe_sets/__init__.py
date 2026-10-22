# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe-set builder (CL13): the frozen evaluation surface for Phase 1.

Prompts are built once per architecture generation and shared across the four
architectures; reference continuations are recorded per architecture from ``B``, the
experts, and the strong teacher (a served-model job, not part of the builder). Every
set ships a manifest with row counts and SHA-256 hashes, and the P_agent bucket
tagger turns recorded trajectories into the KL buckets the analysis reads.
"""

from experiments.post_training.probe_sets.buckets import (
    BUCKET_FINAL_MESSAGE,
    BUCKET_REASONING,
    BUCKET_TOOL_ARGUMENTS,
    BUCKET_TOOL_NAME,
    tag_trajectory,
)
from experiments.post_training.probe_sets.builder import (
    PROBE_SET_NAMES,
    ProbeSetSpec,
    build_probe_sets,
    probe_manifest,
)

__all__ = [
    "BUCKET_FINAL_MESSAGE",
    "BUCKET_REASONING",
    "BUCKET_TOOL_ARGUMENTS",
    "BUCKET_TOOL_NAME",
    "PROBE_SET_NAMES",
    "ProbeSetSpec",
    "build_probe_sets",
    "probe_manifest",
    "tag_trajectory",
]
