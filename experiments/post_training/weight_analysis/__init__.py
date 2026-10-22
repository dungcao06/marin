# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight analysis and grafting over WP2's fp32 exports (CL15 grafting, CL16).

The delta statistics follow the method of Karan Sagar's #9920/#9984 weight-delta
analyses (the referenced branch is unreachable; the published comments are the
spec): per-component relative Frobenius norms of each checkpoint's delta against the
student B in fp32, delta cosines, sign agreement on both-changed weights, and
subspace overlap of one delta's energy inside another's top singular directions
against random and permuted controls. On top of the port sit the Phase 1a additions:
per-expert norms and Gini, per-expert conflict between task deltas, additivity of
joint vs single-task deltas, and seed-ceiling normalization. The grafting builder
writes ``B + Δ·1[S]`` checkpoints restricted to Grug component groups.
"""
