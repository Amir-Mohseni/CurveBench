#!/usr/bin/env python3
"""
Test script for CurveBench environment.

Run this to verify the environment works locally before pushing to Hub.
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / "configs" / "lab" / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        # python-dotenv not available, try manual loading
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

from curvebench_env import load_environment, parse_tree_from_response, are_trees_isomorphic


def test_parsing():
    """Test tree parsing from various formats."""
    print("Testing tree parsing...")
    
    test_cases = [
        ('<answer>\n1 0\n2 0\n3 1\n</answer>', [(0, 1), (0, 2), (1, 3)]),
        ('<answer>1 0\n2 0</answer>', [(0, 1), (0, 2)]),
        ('<answer>\n</answer>', []),
    ]
    
    for text, expected in test_cases:
        result = parse_tree_from_response(text)
        assert result == expected, f"Failed for '{text}': got {result}, expected {expected}"
        print(f"  ✓ Parsed: {text[:50]}... -> {result}")
    
    print("All parsing tests passed!\n")


def test_isomorphism():
    """Test tree isomorphism checking."""
    print("Testing tree isomorphism...")
    
    # Same tree
    tree1 = [(0, 1), (0, 2), (1, 3)]
    tree2 = [(0, 1), (0, 2), (1, 3)]
    assert are_trees_isomorphic(tree1, tree2) == True
    print("  ✓ Same tree")
    
    # Isomorphic (different order)
    tree1 = [(0, 1), (0, 2), (1, 3)]
    tree2 = [(0, 2), (0, 1), (1, 3)]
    assert are_trees_isomorphic(tree1, tree2) == True
    print("  ✓ Isomorphic trees")
    
    # Non-isomorphic
    tree1 = [(0, 1), (0, 2), (0, 3)]  # Star
    tree2 = [(0, 1), (1, 2), (2, 3)]  # Path
    assert are_trees_isomorphic(tree1, tree2) == False
    print("  ✓ Non-isomorphic trees")
    
    print("All isomorphism tests passed!\n")


def test_environment_load():
    """Test loading the environment."""
    print("Testing environment loading...")
    
    try:
        # Try loading with a small split first
        env = load_environment(split="total_train")
        print(f"  ✓ Environment loaded successfully")
        print(f"  ✓ Dataset size: {len(env.dataset)}")
        
        # Check first sample structure
        if len(env.dataset) > 0:
            sample = env.dataset[0]
            assert "prompt" in sample
            assert "answer" in sample
            print(f"  ✓ Sample structure correct")
            print(f"    - Prompt type: {type(sample['prompt'])}")
            print(f"    - Answer type: {type(sample['answer'])}")
            print(f"    - Answer format: <answer>...</answer> with u v per line")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to load environment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CurveBench Environment Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    test_parsing()
    test_isomorphism()
    
    success = test_environment_load()
    
    print("=" * 60)
    if success:
        print("✓ All tests passed! Environment is ready to push.")
    else:
        print("✗ Some tests failed. Please fix issues before pushing.")
        sys.exit(1)
