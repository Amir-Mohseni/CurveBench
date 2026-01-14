"""
Tree isomorphism checking utilities.

This module provides functions to check if two trees (represented as edge lists)
are isomorphic using NetworkX library.
"""

from typing import List, Tuple
import networkx as nx


def are_trees_isomorphic(edges1: List[Tuple[int, int]], 
                         edges2: List[Tuple[int, int]]) -> bool:
    """
    Check if two trees are isomorphic.
    
    Two trees are isomorphic if they have the same structure, meaning
    one can be obtained from the other by relabeling nodes.
    
    Edges are treated as undirected (non-directed graphs).
    
    Args:
        edges1: First tree as list of (u, v) tuples representing edges
        edges2: Second tree as list of (u, v) tuples representing edges
        
    Returns:
        True if trees are isomorphic, False otherwise
        
    Examples:
        >>> edges1 = [(0, 1), (0, 2), (1, 3)]
        >>> edges2 = [(0, 2), (0, 1), (2, 3)]
        >>> are_trees_isomorphic(edges1, edges2)
        True
        
        >>> edges1 = [(0, 1), (0, 2), (0, 3)]  # Star graph
        >>> edges2 = [(0, 1), (1, 2), (2, 3)]  # Path graph
        >>> are_trees_isomorphic(edges1, edges2)
        False
    """
    # Handle empty trees
    if not edges1 and not edges2:
        return True
    if not edges1 or not edges2:
        return False
    
    # Create undirected graphs from edge lists
    G1 = nx.Graph(edges1)
    G2 = nx.Graph(edges2)
    
    # Check if they have the same number of nodes
    if len(G1.nodes) != len(G2.nodes):
        return False
    
    # Check if they have the same number of edges
    if len(G1.edges) != len(G2.edges):
        return False
    
    # Use NetworkX isomorphism checker for undirected graphs
    # For trees, we use is_isomorphic which checks structural isomorphism
    return nx.is_isomorphic(G1, G2)


if __name__ == "__main__":
    # Example usage and tests
    print("Testing tree isomorphism using NetworkX...")
    
    # Test 1: Same tree
    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 1), (0, 2), (1, 3)]
    assert are_trees_isomorphic(edges1, edges2) == True
    print("✓ Test 1 passed: Same tree")
    
    # Test 2: Isomorphic trees (same structure, different child order)
    edges1 = [(0, 1), (0, 2), (1, 3)]
    edges2 = [(0, 2), (0, 1), (1, 3)]
    assert are_trees_isomorphic(edges1, edges2) == True
    print("✓ Test 2 passed: Isomorphic trees (same structure)")
    
    # Test 3: Non-isomorphic trees (different structures)
    # Star graph vs path graph with 4 nodes
    edges1 = [(0, 1), (0, 2), (0, 3)]  # Star: node 0 connected to 1, 2, 3
    edges2 = [(0, 1), (1, 2), (2, 3)]  # Path: 0-1-2-3
    assert are_trees_isomorphic(edges1, edges2) == False
    print("✓ Test 3 passed: Non-isomorphic trees (star vs path)")
    
    # Test 4: Empty trees
    assert are_trees_isomorphic([], []) == True
    print("✓ Test 4 passed: Empty trees")
    
    # Test 5: Different number of nodes
    edges1 = [(0, 1), (0, 2)]
    edges2 = [(0, 1)]
    assert are_trees_isomorphic(edges1, edges2) == False
    print("✓ Test 5 passed: Different number of nodes")
    
    # Test 6: Single node trees
    edges1 = []
    edges2 = []
    assert are_trees_isomorphic(edges1, edges2) == True
    print("✓ Test 6 passed: Single node trees (root only)")
    
    print("\nAll tests passed!")

