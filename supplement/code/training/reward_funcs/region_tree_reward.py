"""
Region Tree Reward Functions for GRPO VLM Training.

This module contains reward functions for training VLMs on:
1. Region counting in images of non-crossing closed curves
2. Region-adjacency tree generation from such images
"""

import re
from typing import Optional, List, Tuple

import networkx as nx

from .base import BaseRewardFunction


def extract_answer_content(content: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_answer_format(text: str) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    """
    Parse <answer>...</answer> format (CurveBench).
    First line: number of nodes (excluding root).
    Subsequent lines: "u v" (edge from v to u; v is parent, u is child).
    Returns (num_nodes, list of (parent, child) tuples), or None if parsing fails.
    """
    inner = extract_answer_content(text)
    if not inner:
        return None
    lines = [l.strip() for l in inner.split("\n") if l.strip()]
    if not lines:
        return None
    try:
        num_nodes = int(lines[0])
    except ValueError:
        return None
    edges = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                u, v = int(parts[0]), int(parts[1])
                edges.append((v, u))
            except ValueError:
                continue
    return (num_nodes, edges)


def parse_tree_from_completion(content: str) -> list[tuple[int, int]]:
    """Parse the tree edge list from the model's completion."""
    parsed = parse_answer_format(content)
    if parsed is None:
        return []
    _, edges = parsed
    return edges


def parse_num_nodes_from_completion(content: str) -> Optional[int]:
    """Parse the number of nodes (excluding root) from the model's completion."""
    parsed = parse_answer_format(content)
    if parsed is None:
        return None
    num_nodes, _ = parsed
    return num_nodes


def are_trees_isomorphic(
    edges1: List[Tuple[int, int]], edges2: List[Tuple[int, int]]
) -> bool:
    """Check if two trees are isomorphic using NetworkX."""
    if not edges1 and not edges2:
        return True
    if not edges1 or not edges2:
        return False
    G1 = nx.Graph(edges1)
    G2 = nx.Graph(edges2)
    if len(G1.nodes) != len(G2.nodes) or len(G1.edges) != len(G2.edges):
        return False
    return nx.is_isomorphic(G1, G2)


def tree_correctness_reward(
    completions, tree: list[list[int]], **kwargs
) -> list[float]:
    """
    Reward 1.0 if the predicted tree is isomorphic to the ground truth, else 0.0.

    Args:
        completions: List of model completions; each is a list with one dict
                     containing a 'content' key.
        tree: List of ground-truth edge lists, each like [[0,1],[1,2]].
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, gt_tree in zip(contents, tree):
        try:
            pred_edges = parse_tree_from_completion(content)
            if not pred_edges and gt_tree:
                rewards.append(0.0)
                continue
            gt_edges = [(edge[0], edge[1]) for edge in gt_tree]
            rewards.append(1.0 if are_trees_isomorphic(pred_edges, gt_edges) else 0.0)
        except Exception as e:
            print(f"Tree parsing error: {e}, content: {content[:200]}...")
            rewards.append(0.0)
    return rewards


def num_nodes_reward(
    completions, num_nodes: list[int], **kwargs
) -> list[float]:
    """
    Reward 1.0 if the predicted node count matches the ground truth, else 0.0.

    Args:
        completions: List of model completions.
        num_nodes: List of ground-truth node counts.
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, gt_num in zip(contents, num_nodes):
        try:
            pred_num = parse_num_nodes_from_completion(content)
            if pred_num is None:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if pred_num == gt_num else 0.0)
        except Exception as e:
            print(f"Num nodes parsing error: {e}, content: {content[:200]}...")
            rewards.append(0.0)
    return rewards


def think_answer_format_reward(
    completions: list[list[dict[str, str]]], **kwargs
) -> list[float]:
    """
    Reward 1.0 if the response follows the <think>...</think><answer>...</answer> format.
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>\s*<answer>(.*?)</answer>\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


class TreeCorrectnessReward(BaseRewardFunction):
    """Reward class for tree isomorphism correctness."""

    def __init__(self):
        super().__init__()
        self.__name__ = "TreeCorrectnessReward"

    def calculate_rewards(
        self, completions, tree: list[list[int]] = None, **kwargs
    ) -> list[float]:
        if tree is None:
            tree = kwargs.get("tree", [])
        return tree_correctness_reward(completions, tree, **kwargs)


class NumNodesReward(BaseRewardFunction):
    """Reward class for node-count correctness."""

    def __init__(self):
        super().__init__()
        self.__name__ = "NumNodesReward"

    def calculate_rewards(
        self, completions, num_nodes: list[int] = None, **kwargs
    ) -> list[float]:
        if num_nodes is None:
            num_nodes = kwargs.get("num_nodes", [])
        return num_nodes_reward(completions, num_nodes, **kwargs)


class ThinkAnswerFormatReward(BaseRewardFunction):
    """Reward class for <think>+<answer> format compliance."""

    def __init__(self):
        super().__init__()
        self.__name__ = "ThinkAnswerFormatReward"

    def calculate_rewards(self, completions, **kwargs) -> list[float]:
        return think_answer_format_reward(completions, **kwargs)
