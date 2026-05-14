"""
CurveBench Environment for Prime Intellect Environments Hub.

This environment evaluates vision-language models on their ability to extract
hierarchical tree structures from images containing nested regions.
"""

import base64
import io
import re
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import verifiers as vf
from datasets import load_dataset
from PIL import Image


def are_trees_isomorphic(edges1: List[Tuple[int, int]], 
                         edges2: List[Tuple[int, int]]) -> bool:
    """
    Check if two trees are isomorphic using NetworkX.
    
    Two trees are isomorphic if they have the same structure, meaning
    one can be obtained from the other by relabeling nodes.
    """
    # Handle empty trees
    if not edges1 and not edges2:
        return True
    if not edges1 or not edges2:
        return False
    
    # Create undirected graphs from edge lists
    G1 = nx.Graph(edges1)
    G2 = nx.Graph(edges2)
    
    # Check if they have the same number of nodes and edges
    if len(G1.nodes) != len(G2.nodes) or len(G1.edges) != len(G2.edges):
        return False
    
    # Use NetworkX isomorphism checker
    return nx.is_isomorphic(G1, G2)


def image_to_data_url(image: Image.Image) -> str:
    """Convert PIL Image to data URL."""
    buffered = io.BytesIO()
    # Convert to RGB if needed (some images might be RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


def tree_to_answer_format(edges: List[Tuple[int, int]], num_nodes: int) -> str:
    """
    Convert tree edges to <answer>...</answer> format.
    First line: number of nodes (excluding root).
    Subsequent lines: "u v" meaning edge from v to u (v is parent, u is child).
    Internal format is (parent, child), so (v, u) -> line "u v".
    """
    if not edges:
        return "<answer>\n0\n</answer>"
    edge_lines = [f"{child} {parent}" for parent, child in edges]
    return "<answer>\n" + str(num_nodes) + "\n" + "\n".join(edge_lines) + "\n</answer>"


def parse_answer_format(text: str) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    """
    Parse <answer>...</answer> format.
    First line: number of nodes (excluding root).
    Subsequent lines: "u v" (edge from v to u).
    Returns (num_nodes, list of (parent, child) tuples), or None if parsing fails.
    """
    match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    inner = match.group(1).strip()
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
            u, v = int(parts[0]), int(parts[1])
            edges.append((v, u))  # edge from v to u -> (parent=v, child=u)
    return (num_nodes, edges)


def parse_tree_from_response(text: str) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    """
    Parse tree structure from model response.
    Returns (num_nodes, list of (parent, child) tuples), or None if parsing fails.
    """
    if not text:
        return None
    return parse_answer_format(text)


def create_prompt(image_data_url: str) -> List[Dict[str, Any]]:
    """
    Create a prompt for the model.
    
    Args:
        image_data_url: Data URL of the image
    
    Returns:
        List of messages in OpenAI-compatible format
    """
    instruction = """Analyze this image and extract the hierarchical tree structure representing the nested regions.

The image contains nested shapes/regions. Your task is to identify the parent-child relationships between these regions.

Return the tree structure as a list of edges, where each edge is represented as (parent, child).
- The root node is always 0
- Each region is assigned a unique node number
- Edges represent parent-child relationships (a parent region contains a child region)

Format your response inside <answer>...</answer> tags.
- The first line should be the number of nodes (excluding the root).
- Each subsequent line should be "u v" meaning an edge from v to u (v is the parent, u is the child).

Example:
<answer>
3
1 0
2 0
3 1
</answer>

Make sure to include ALL edges that represent the hierarchical structure."""
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                }
            ]
        }
    ]


def _parse_ground_truth(answer: Any) -> Tuple[int, List[Tuple[int, int]]]:
    """Parse ground truth answer into (num_nodes, list of (parent, child) edge tuples)."""
    if isinstance(answer, str):
        parsed = parse_answer_format(answer)
        if parsed is not None:
            return parsed
    return (0, [])


def _extract_response_text(completion: List[Dict[str, Any]]) -> Optional[str]:
    """Extract text content from the last message in a completion."""
    if not completion:
        return None
    last_message = completion[-1]
    content = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        content = " ".join(text_parts)
    return str(content)


def tree_reward(completion: List[Dict[str, Any]], answer: Any) -> float:
    """Reward 1.0 if the predicted tree is isomorphic to the ground truth."""
    try:
        gt_num_nodes, gt_edges = _parse_ground_truth(answer)
        content = _extract_response_text(completion)
        if content is None:
            return 0.0
        parsed = parse_tree_from_response(content)
        if parsed is None:
            return 0.0
        _, pred_edges = parsed
        return 1.0 if are_trees_isomorphic(pred_edges, gt_edges) else 0.0
    except Exception as e:
        print(f"Error in tree_reward: {e}")
        return 0.0


def node_count_reward(completion: List[Dict[str, Any]], answer: Any) -> float:
    """Reward 1.0 if the predicted node count matches the ground truth."""
    try:
        gt_num_nodes, _ = _parse_ground_truth(answer)
        content = _extract_response_text(completion)
        if content is None:
            return 0.0
        parsed = parse_tree_from_response(content)
        if parsed is None:
            return 0.0
        pred_num_nodes, _ = parsed
        return 1.0 if pred_num_nodes == gt_num_nodes else 0.0
    except Exception as e:
        print(f"Error in node_count_reward: {e}")
        return 0.0


def load_environment(split: str = "total_train") -> vf.Environment:
    """
    Load the CurveBench environment.
    
    Args:
        split: Dataset split to use (default: "total_train")
               Options: "total_train", "total_validation", "total_test",
                        "level_1_train", "level_2_train", "level_3_train", etc.
    
    Returns:
        A verifiers Environment (SingleTurnEnv) configured for CurveBench
    """
    # Load dataset from HuggingFace Hub
    try:
        dataset = load_dataset("AmirMohseni/CurveBench-Easy", split=split)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset split '{split}'. "
            f"Available splits: total_train, total_validation, total_test, "
            f"level_1_train, level_1_validation, level_1_test, etc. "
            f"Error: {e}"
        )
    
    # Convert dataset to verifiers format
    def transform_row(row):
        """Transform a dataset row to verifiers format."""
        # Convert PIL Image to data URL
        image_data_url = image_to_data_url(row["image"])
        
        # Create prompt
        prompt = create_prompt(image_data_url)
        
        tree = row["tree"]
        num_nodes = row.get("num_nodes", 0)
        if isinstance(tree, list) and len(tree) > 0:
            if isinstance(tree[0], list):
                tree = [tuple(edge) for edge in tree]
            answer = tree_to_answer_format(tree, num_nodes)
            tree_list = tree
        else:
            answer = "<answer>\n0\n</answer>"
            tree_list = []
        
        return {
            "prompt": prompt,
            "answer": answer,
            "info": {
                "num_nodes": row.get("num_nodes"),
                "tree": tree_list
            }
        }
    
    # Transform dataset
    verifiers_dataset = dataset.map(transform_row, remove_columns=["image"])
    
    rubric = vf.Rubric(
        funcs=[tree_reward, node_count_reward],
        weights=[0.7, 0.3]
    )
    
    # Create and return SingleTurnEnv
    env = vf.SingleTurnEnv(
        dataset=verifiers_dataset,
        rubric=rubric
    )
    
    return env
