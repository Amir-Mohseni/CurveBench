"""
Tree isomorphism checking utilities.

This module provides functions to check if two trees (represented as edge lists)
are isomorphic.
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict


def build_tree_from_edges(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Build an adjacency list representation of a tree from edge list.
    
    Args:
        edges: List of (parent, child) tuples
        
    Returns:
        Dictionary mapping each node to its list of children
    """
    tree = defaultdict(list)
    all_nodes = set()
    
    for parent, child in edges:
        tree[parent].append(child)
        all_nodes.add(parent)
        all_nodes.add(child)
    
    # Ensure all nodes are in the tree dict (even if they have no children)
    for node in all_nodes:
        if node not in tree:
            tree[node] = []
    
    return dict(tree)


def get_canonical_form(tree: Dict[int, List[int]], root: int) -> str:
    """
    Get a canonical string representation of a rooted tree.
    
    This recursively builds a canonical form by sorting children
    and concatenating their canonical forms.
    
    Args:
        tree: Adjacency list representation of the tree
        root: Root node of the tree
        
    Returns:
        Canonical string representation
    """
    children = sorted(tree.get(root, []))
    
    if not children:
        return "()"
    
    child_forms = [get_canonical_form(tree, child) for child in children]
    # Sort child forms to ensure canonical ordering
    child_forms.sort()
    
    return "(" + "".join(child_forms) + ")"


def find_root(edges: List[Tuple[int, int]]) -> int:
    """
    Find the root node of a tree from its edge list.
    
    The root is the node that appears as a parent but never as a child,
    or if all nodes have parents, it's the node with the smallest index
    that appears as a parent.
    
    Args:
        edges: List of (parent, child) tuples
        
    Returns:
        Root node index
    """
    if not edges:
        return 0
    
    parents = set()
    children = set()
    
    for parent, child in edges:
        parents.add(parent)
        children.add(child)
    
    # Root is a parent that is not a child
    roots = parents - children
    
    if roots:
        return min(roots)
    
    # If all nodes have parents, return the smallest parent
    return min(parents) if parents else 0


def are_trees_isomorphic(edges1: List[Tuple[int, int]], 
                         edges2: List[Tuple[int, int]]) -> bool:
    """
    Check if two trees are isomorphic.
    
    Two trees are isomorphic if they have the same structure, meaning
    one can be obtained from the other by relabeling nodes.
    
    Args:
        edges1: First tree as list of (parent, child) tuples
        edges2: Second tree as list of (parent, child) tuples
        
    Returns:
        True if trees are isomorphic, False otherwise
        
    Examples:
        >>> edges1 = [(0, 1), (0, 2), (1, 3)]
        >>> edges2 = [(0, 2), (0, 1), (2, 3)]
        >>> are_trees_isomorphic(edges1, edges2)
        True
        
        >>> edges1 = [(0, 1), (0, 2)]
        >>> edges2 = [(0, 1), (1, 2)]
        >>> are_trees_isomorphic(edges1, edges2)
        False
    """
    # Handle empty trees
    if not edges1 and not edges2:
        return True
    if not edges1 or not edges2:
        return False
    
    # Check if they have the same number of nodes
    nodes1 = set()
    nodes2 = set()
    for p, c in edges1:
        nodes1.add(p)
        nodes1.add(c)
    for p, c in edges2:
        nodes2.add(p)
        nodes2.add(c)
    
    if len(nodes1) != len(nodes2):
        return False
    
    # Build tree structures
    tree1 = build_tree_from_edges(edges1)
    tree2 = build_tree_from_edges(edges2)
    
    # Find roots
    root1 = find_root(edges1)
    root2 = find_root(edges2)
    
    # Get canonical forms
    canonical1 = get_canonical_form(tree1, root1)
    canonical2 = get_canonical_form(tree2, root2)
    
    return canonical1 == canonical2


if __name__ == "__main__":
    # Example usage and tests
    print("Testing tree isomorphism...")
    
    # Test 1: Same tree
    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 1), (0, 2), (1, 3)]
    assert are_trees_isomorphic(edges1, edges2) == True
    print("✓ Test 1 passed: Same tree")
    
    # Test 2: Isomorphic trees (different node labels)
    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 2), (0, 1), (2, 3)]
    assert are_trees_isomorphic(edges1, edges2) == True
    print("✓ Test 2 passed: Isomorphic trees")
    
    # Test 3: Non-isomorphic trees
    edges1 = [(0, 1), (0, 2)]
    edges2 = [(0, 1), (1, 2)]
    assert are_trees_isomorphic(edges1, edges2) == False
    print("✓ Test 3 passed: Non-isomorphic trees")
    
    # Test 4: Empty trees
    assert are_trees_isomorphic([], []) == True
    print("✓ Test 4 passed: Empty trees")
    
    # Test 5: Different number of nodes
    edges1 = [(0, 1), (0, 2)]
    edges2 = [(0, 1)]
    assert are_trees_isomorphic(edges1, edges2) == False
    print("✓ Test 5 passed: Different number of nodes")
    
    print("\nAll tests passed!")

