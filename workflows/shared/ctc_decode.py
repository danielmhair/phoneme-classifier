"""
Shared CTC decoding utilities.

Single source of truth for turning a CTC model's per-frame log-probabilities
into (a) a top-1 predicted phoneme and (b) a per-class score vector suitable
for ranking/confidence-margin reporting.

Before this module existed, three different decode algorithms were live
simultaneously across the codebase:
  1. Correct greedy CTC decode (collapse repeats, drop blank) - used only for
     in-process PyTorch training-time validation (CTCModel.predict()).
  2. "First non-blank timestep" decode - used in s7_test_onnx.py.
  3. np.mean(log_probs, axis=0) then softmax - used by inference/cli/model_loader.py
     (and therefore by the CLI, temporal brain, and evaluation/model_comparison_*.py).
     This is the "broken decoding" bug documented in the PRD: averaging log-probs
     across time destroys CTC's frame-by-frame alignment structure.

This module replaces all three with one implementation, used everywhere.
"""
from typing import List, Optional, Tuple

import numpy as np


def greedy_ctc_decode(log_probs: np.ndarray, blank_id: Optional[int] = None) -> List[int]:
    """Greedy CTC decode: argmax per frame, collapse consecutive repeats, drop blanks.

    Args:
        log_probs: [T, C] log-probabilities (or raw logits - only relative
            order within each frame matters for argmax).
        blank_id: index of the CTC blank class. Defaults to C-1 (last class),
            matching this project's CTCHead convention
            (workflows/*/models/ctc_model.py: blank_token = num_classes - 1).

    Returns:
        Collapsed sequence of class indices (blank removed, consecutive
        duplicate frame-predictions collapsed to one).
    """
    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, C] log_probs, got shape {log_probs.shape}")
    num_classes = log_probs.shape[1]
    if blank_id is None:
        blank_id = num_classes - 1

    frame_preds = np.argmax(log_probs, axis=-1)
    decoded: List[int] = []
    prev = None
    for token in frame_preds:
        token = int(token)
        if token != blank_id and token != prev:
            decoded.append(token)
        prev = token
    return decoded


def ctc_class_scores(log_probs: np.ndarray, blank_id: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
    """Per-class score = peak (max-over-time) log-prob, excluding the blank class.

    Time-averaging (the old bug) destroys CTC's alignment structure - it asks
    "what did this clip sound like on average across every frame, including
    silence/blank-dominated ones," which is the wrong question for CTC output.
    Peak-posterior scoring instead asks "did this class ever appear
    confidently at some frame," which is the natural formulation for a short,
    isolated single-phoneme clip where the target should peak strongly at one
    point in the sequence.

    Returns:
        (scores, class_indices) where scores[i] is the peak log-prob for
        class_indices[i]. class_indices is every class except blank_id, in
        ascending order - for this project's convention (blank = last index)
        that is exactly range(len(phoneme_labels)), i.e. already aligned to
        phoneme_labels.json order.
    """
    num_classes = log_probs.shape[1]
    if blank_id is None:
        blank_id = num_classes - 1
    class_indices = [c for c in range(num_classes) if c != blank_id]
    scores = log_probs[:, class_indices].max(axis=0)
    return scores, class_indices


def ctc_predict(log_probs: np.ndarray, blank_id: Optional[int] = None) -> Tuple[int, np.ndarray, List[int]]:
    """Combine greedy decode (primary label) with peak-posterior scores (ranking/margin).

    Args:
        log_probs: [T, C] log-probabilities for one clip.
        blank_id: see greedy_ctc_decode().

    Returns:
        top1_class_index: the predicted phoneme class index.
            - If greedy decode collapses to exactly one token, that's the answer.
            - If it collapses to more than one distinct token (model produced a
              noisy/ambiguous sequence), tie-break by picking whichever of the
              decoded classes has the highest peak posterior.
            - If every frame decoded to blank (no confident prediction at all),
              fall back to the peak-posterior argmax over all non-blank classes.
        probabilities: softmax over peak-posterior scores, indexed by
            class_indices (see ctc_class_scores) - i.e. aligned to
            phoneme_labels.json order under this project's blank-is-last convention.
        decoded_sequence: the raw greedy-decoded token sequence, kept for
            diagnostics (e.g. flagging clips where greedy decode was ambiguous).
    """
    decoded = greedy_ctc_decode(log_probs, blank_id)
    scores, class_indices = ctc_class_scores(log_probs, blank_id)

    exp_scores = np.exp(scores - scores.max())
    probabilities = exp_scores / exp_scores.sum()

    if len(decoded) == 1:
        top1 = decoded[0]
    elif len(decoded) > 1:
        decoded_set = set(decoded)
        candidate_scores = {c: scores[class_indices.index(c)] for c in decoded_set}
        top1 = max(candidate_scores, key=candidate_scores.get)
    else:
        top1 = class_indices[int(np.argmax(scores))]

    return top1, probabilities, decoded
