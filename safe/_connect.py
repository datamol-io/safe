"""Connectivity tracking for SAFE completion decoding.

Dependency-free helpers that decide, from the SAFE tokens generated so far,
whether a scaffold completion is already a single connected molecule. The
:class:`safe.sample.ScaffoldConnectivityLogitsProcessor` uses this to steer
generation so that completion tasks (scaffold decoration, motif extension and
super-structure) return one connected molecule instead of the scaffold plus
spurious disconnected fragments.

The logic here is intentionally torch-free so it can be unit tested in the core
matrix without the model stack.
"""

from typing import List, NamedTuple, Optional

_DIGITS = frozenset("0123456789")


def ring_label(token: str) -> Optional[int]:
    """Return the ring-closure label an atom-glue token encodes, else ``None``.

    Handles bare single digits, the two-digit ``%NN`` form and RDKit's extended
    ``%(nnn)`` form. Bracket atoms (``[13C]``, ``[NH3+]``, ``[*:12]``) are single
    tokens whose inner digits are never ring closures, so they return ``None``.
    """
    if len(token) == 1 and token in _DIGITS:
        return int(token)
    if token.startswith("%(") and token.endswith(")"):
        inner = token[2:-1]
        return int(inner) if inner.isdigit() else None
    if token.startswith("%") and token[1:].isdigit():
        return int(token[1:])
    return None


class Decision(NamedTuple):
    """Outcome of :func:`analyze` for one partially generated sequence."""

    complete: bool  # single connected component, balanced, no open labels
    current_attached: bool  # the fragment being written is joined to the scaffold
    started_completion: bool  # at least one non-special token was generated
    open_label_count: int
    connected: bool  # all fragments belong to one component (via closed labels)
    balanced: bool  # parentheses are balanced


def analyze(
    prefix_tokens: List[str],
    generated_tokens: List[str],
    special_tokens: Optional[frozenset] = None,
) -> Decision:
    """Assess the connectivity of ``prefix_tokens + generated_tokens``.

    The prefix is the encoded scaffold (a single connected molecule, possibly
    written as several ``.``-separated SAFE fragments joined by ring closures).
    Fragments are unioned as ring-closure labels pair up, so a decoration
    "attaches" to the scaffold when it closes one of the scaffold's open labels.

    Args:
        prefix_tokens: SAFE tokens of the scaffold prompt (specials removed or not).
        generated_tokens: SAFE tokens produced after the prompt.
        special_tokens: token strings to ignore (BOS/EOS/PAD and friends).
    """
    specials = special_tokens or frozenset()
    tokens = prefix_tokens + generated_tokens
    n_prefix = len(prefix_tokens)

    parent = {0: 0}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    frag = 0
    open_labels = {}  # label -> fragment index that currently holds it open
    generated_real = 0
    paren_depth = 0

    for i, tok in enumerate(tokens):
        if tok in specials:
            continue
        is_generated = i >= n_prefix
        if tok == "(":
            paren_depth += 1
            continue
        if tok == ")":
            paren_depth = max(0, paren_depth - 1)
            continue
        if is_generated:
            generated_real += 1
        if tok == ".":
            frag += 1
            parent.setdefault(frag, frag)
            continue
        label = ring_label(tok)
        if label is not None:
            if label in open_labels:
                union(frag, open_labels.pop(label))
            else:
                open_labels[label] = frag

    scaffold_root = find(0)
    connected = all(find(f) == scaffold_root for f in parent)
    has_open = bool(open_labels)
    balanced = paren_depth == 0
    started = generated_real > 0
    complete = connected and not has_open and balanced and started
    current_attached = find(frag) == scaffold_root

    return Decision(
        complete=complete,
        current_attached=current_attached,
        started_completion=started,
        open_label_count=len(open_labels),
        connected=connected,
        balanced=balanced,
    )
