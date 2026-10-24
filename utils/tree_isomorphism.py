"""Rooted tree isomorphism for CurveBench containment trees.

Edges are directed ``(parent, child)`` with the exterior region fixed at root
``0``. Sibling order is ignored; undirected shape matching is not enough,
because two trees can share the same unlabeled graph while encoding different
containment hierarchies.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Edge = Tuple[int, int]
Signature = Tuple["Signature", ...]


def _children_map(edges: Iterable[Edge]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {0: []}
    for parent, child in edges:
        children.setdefault(parent, []).append(child)
        children.setdefault(child, [])
    return children


def _is_valid_rooted_tree(edges: Sequence[Edge]) -> bool:
    """Validate a directed tree rooted at node 0."""
    if any(parent < 0 or child < 0 for parent, child in edges):
        return False
    if len(set(edges)) != len(edges):
        return False
    if any(parent == child for parent, child in edges):
        return False
    if any(child == 0 for _parent, child in edges):
        return False

    parents: Dict[int, int] = {}
    nodes = {0}
    for parent, child in edges:
        if child in parents:
            return False
        parents[child] = parent
        nodes.update((parent, child))

    non_root = nodes - {0}
    if len(edges) != len(non_root):
        return False
    if any(node not in parents for node in non_root):
        return False

    children = _children_map(edges)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            return False
        visited.add(node)
        stack.extend(children.get(node, []))
    return visited == nodes


def canonical_signature(
    edges: Sequence[Edge],
    max_depth: Optional[int] = None,
) -> Optional[Signature]:
    """AHU-style signature for an unordered tree rooted at 0.

    Returns ``None`` when ``edges`` is not a valid rooted tree.
    """
    normalized = [(int(parent), int(child)) for parent, child in edges]
    if not _is_valid_rooted_tree(normalized):
        return None

    children = _children_map(normalized)

    def visit(node: int, depth: int) -> Signature:
        if max_depth is not None and depth >= max_depth:
            return ()
        return tuple(sorted(visit(child, depth + 1) for child in children[node]))

    return visit(0, 0)


def are_trees_isomorphic(
    edges1: List[Tuple[int, int]],
    edges2: List[Tuple[int, int]],
) -> bool:
    """
    Check if two rooted containment trees are isomorphic.

    Node ``0`` is fixed as the exterior root. Non-root labels may differ and
    sibling order is ignored. Invalid trees never match, including against
    another invalid tree.

    Args:
        edges1: First tree as list of ``(parent, child)`` edges
        edges2: Second tree as list of ``(parent, child)`` edges

    Returns:
        True if the trees are rooted-isomorphic, False otherwise

    Examples:
        >>> edges1 = [(0, 1), (0, 2), (1, 3)]
        >>> edges2 = [(0, 2), (0, 1), (2, 3)]
        >>> are_trees_isomorphic(edges1, edges2)
        True

        >>> edges1 = [(0, 1), (0, 2), (0, 3)]  # Star rooted at 0
        >>> edges2 = [(0, 1), (1, 2), (2, 3)]  # Path rooted at 0
        >>> are_trees_isomorphic(edges1, edges2)
        False

        # Same undirected shape, different root / containment hierarchy:
        >>> edges1 = [(0, 1), (1, 2), (2, 3)]
        >>> edges2 = [(0, 1), (0, 3), (1, 2)]
        >>> are_trees_isomorphic(edges1, edges2)
        False
    """
    signature1 = canonical_signature(edges1)
    signature2 = canonical_signature(edges2)
    if signature1 is None or signature2 is None:
        return False
    return signature1 == signature2


if __name__ == "__main__":
    print("Testing rooted tree isomorphism...")

    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 1), (0, 2), (1, 3)]
    assert are_trees_isomorphic(edges1, edges2) is True
    print("✓ Test 1 passed: Same tree")

    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 2), (0, 1), (2, 3)]
    assert are_trees_isomorphic(edges1, edges2) is True
    print("✓ Test 2 passed: Sibling / child-label permutation")

    edges1 = [(0, 1), (0, 2), (0, 3)]
    edges2 = [(0, 1), (1, 2), (2, 3)]
    assert are_trees_isomorphic(edges1, edges2) is False
    print("✓ Test 3 passed: Star vs path")

    assert are_trees_isomorphic([], []) is True
    print("✓ Test 4 passed: Empty / root-only trees")

    # Undirected-isomorphism false positive: path vs bushier tree of same size.
    edges1 = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 6), (4, 5)]
    edges2 = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    assert are_trees_isomorphic(edges1, edges2) is False
    print("✓ Test 5 passed: Rejects undirected false positive")

    edges1 = [(0, 1), (0, 2)]
    edges2 = [(0, 1)]
    assert are_trees_isomorphic(edges1, edges2) is False
    print("✓ Test 6 passed: Different number of nodes")

    print("\nAll tests passed!")
