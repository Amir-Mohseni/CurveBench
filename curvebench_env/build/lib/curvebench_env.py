"""
CurveBench Environment for Prime Intellect Environments Hub.

This environment evaluates vision-language models on their ability to extract
hierarchical tree structures from images containing nested regions.
"""

import base64
import io
import json
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


def parse_tree_from_response(text: str) -> Optional[List[Tuple[int, int]]]:
    """
    Parse tree structure from model response.
    
    Supports multiple formats:
    1. JSON: {"tree": [[0,1], [0,2]]} or {"edges": [[0,1], [0,2]]}
    2. Simple format: TREE: (0,1) (0,2) or EDGES: (0,1) (0,2)
    3. List format: [[0,1], [0,2], [1,3]]
    4. Tuple format: [(0,1), (0,2), (1,3)]
    
    Returns:
        List of (parent, child) tuples, or None if parsing fails
    """
    if not text:
        return None
    
    # Try JSON format first
    json_patterns = [
        r'\{[^}]*"tree"\s*:\s*\[([^\]]+)\][^}]*\}',
        r'\{[^}]*"edges"\s*:\s*\[([^\]]+)\][^}]*\}',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # Try to extract the full JSON
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    tree = data.get('tree') or data.get('edges') or data.get('tree_edges')
                    if tree:
                        return [(int(edge[0]), int(edge[1])) for edge in tree]
            except:
                pass
    
    # Try to find JSON array directly
    json_array_match = re.search(r'\[\[[\d\s,]+\]\]', text)
    if json_array_match:
        try:
            edges = json.loads(json_array_match.group())
            return [(int(edge[0]), int(edge[1])) for edge in edges]
        except:
            pass
    
    # Try simple format: TREE: (0,1) (0,2) or EDGES: (0,1) (0,2)
    tree_pattern = r'(?:TREE|EDGES|tree|edges)[:\s]+(\([\d\s,]+\))'
    match = re.search(tree_pattern, text, re.IGNORECASE)
    if match:
        # Extract all (parent, child) tuples
        tuples = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
        if tuples:
            return [(int(p), int(c)) for p, c in tuples]
    
    # Try to find all (parent, child) tuples anywhere in the text
    tuples = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
    if tuples:
        return [(int(p), int(c)) for p, c in tuples]
    
    # Try bracket format: [0,1] [0,2]
    brackets = re.findall(r'\[(\d+)\s*,\s*(\d+)\]', text)
    if brackets:
        return [(int(p), int(c)) for p, c in brackets]
    
    return None


def create_prompt(image_data_url: str, num_nodes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Create a prompt for the model.
    
    Args:
        image_data_url: Data URL of the image
        num_nodes: Optional number of nodes (for context)
    
    Returns:
        List of messages in OpenAI-compatible format
    """
    instruction = """Analyze this image and extract the hierarchical tree structure representing the nested regions.

The image contains nested shapes/regions. Your task is to identify the parent-child relationships between these regions.

Return the tree structure as a list of edges, where each edge is represented as (parent, child).
- The root node is always 0
- Each region is assigned a unique node number
- Edges represent parent-child relationships (a parent region contains a child region)

Format your response as JSON with a "tree" field containing the list of edges:
{"tree": [[0, 1], [0, 2], [1, 3]]}

Or use a simple format:
TREE: (0,1) (0,2) (1,3)

Make sure to include ALL edges that represent the hierarchical structure."""
    
    if num_nodes is not None:
        instruction += f"\n\nNote: The tree should have approximately {num_nodes} nodes (excluding the root)."
    
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


def tree_reward(completion: List[Dict[str, Any]], answer: List[Tuple[int, int]], 
                info: Optional[Dict[str, Any]] = None) -> float:
    """
    Reward function that compares predicted tree to ground truth using isomorphism.
    
    Args:
        completion: List of messages from the model (OpenAI format)
        answer: Ground truth tree as list of (parent, child) edges
        info: Optional additional info (e.g., num_nodes)
    
    Returns:
        Reward score: 1.0 if trees are isomorphic, 0.0 otherwise
    """
    try:
        # Extract the final message content
        if not completion:
            return 0.0
        
        # Get the last message's content
        last_message = completion[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)
        
        # If content is a list (multimodal), extract text
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            content = " ".join(text_parts)
        
        # Parse tree from response
        predicted_tree = parse_tree_from_response(str(content))
        
        if predicted_tree is None:
            return 0.0
        
        # Compare trees using isomorphism
        if are_trees_isomorphic(predicted_tree, answer):
            return 1.0
        else:
            return 0.0
            
    except Exception as e:
        # Log error but return 0.0
        print(f"Error in tree_reward: {e}")
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
        prompt = create_prompt(image_data_url, num_nodes=row.get("num_nodes"))
        
        # Ground truth answer (tree edges)
        # Ensure tree is a list of tuples
        tree = row["tree"]
        if isinstance(tree, list) and len(tree) > 0:
            # Convert nested lists to tuples if needed
            if isinstance(tree[0], list):
                tree = [tuple(edge) for edge in tree]
            answer = tree
        else:
            answer = []
        
        return {
            "prompt": prompt,
            "answer": answer,
            "info": {
                "num_nodes": row.get("num_nodes"),
                "tree": answer
            }
        }
    
    # Transform dataset
    verifiers_dataset = dataset.map(transform_row, remove_columns=["image"])
    
    # Create rubric with tree comparison reward
    rubric = vf.Rubric(
        [tree_reward]
    )
    
    # Create and return SingleTurnEnv
    env = vf.SingleTurnEnv(
        dataset=verifiers_dataset,
        rubric=rubric
    )
    
    return env
