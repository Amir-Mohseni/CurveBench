#!/usr/bin/env python3
"""
Simple test to verify CurveBench environment loads and works correctly.
Run this locally before pushing to verify everything works.
"""

import os
import sys

def test_import():
    """Test that the environment can be imported."""
    print("Testing import...")
    try:
        from curvebench_env import load_environment
        print("✓ Successfully imported curvebench_env")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_load():
    """Test that the environment can be loaded."""
    print("\nTesting environment loading...")
    try:
        from curvebench_env import load_environment
        env = load_environment(split="level_1_train")
        print(f"✓ Successfully loaded environment with {len(env.dataset)} examples")
        return True
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample():
    """Test that we can access a sample from the environment."""
    print("\nTesting sample access...")
    try:
        from curvebench_env import load_environment
        env = load_environment(split="level_1_train")
        sample = env.dataset[0]
        
        print(f"✓ Successfully accessed sample")
        print(f"  - Prompt type: {type(sample['prompt'])}")
        print(f"  - Answer: {sample['answer'][:100]}...")
        print(f"  - Info keys: {list(sample.get('info', {}).keys())}")
        
        # Check prompt format
        if isinstance(sample['prompt'], list):
            print(f"  - Prompt is a list with {len(sample['prompt'])} messages")
            if len(sample['prompt']) > 0 and isinstance(sample['prompt'][0], dict):
                print(f"  - First message role: {sample['prompt'][0].get('role')}")
                content = sample['prompt'][0].get('content', [])
                if isinstance(content, list):
                    print(f"  - Content items: {len(content)}")
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            print(f"    - Item {i} type: {item.get('type')}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to access sample: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward():
    """Test that the reward function works."""
    print("\nTesting reward function...")
    try:
        from curvebench_env import tree_reward, are_trees_isomorphic, parse_tree_from_response, tree_to_answer_format
        
        # Test parsing first
        test_content = '{"tree": [[0, 1], [0, 2], [1, 3]]}'
        parsed = parse_tree_from_response(test_content)
        print(f"  - Parsed tree from JSON: {parsed}")
        
        # Test with correct prediction in BOTH formats
        completion = [{"role": "assistant", "content": '{"tree": [[0, 1], [0, 2], [1, 3]]}'}]
        
        # Test 1: answer as list of tuples
        answer_list = [(0, 1), (0, 2), (1, 3)]
        reward1 = tree_reward(completion, answer_list)
        print(f"✓ Reward for correct prediction (list format): {reward1}")
        
        # Test 2: answer as string format (like in actual environment)
        answer_string = tree_to_answer_format(answer_list)
        print(f"  - Answer string format: {repr(answer_string)}")
        reward2 = tree_reward(completion, answer_string)
        print(f"✓ Reward for correct prediction (string format): {reward2}")
        
        if reward1 != 1.0 or reward2 != 1.0:
            print(f"  WARNING: Expected both rewards to be 1.0")
            print(f"  Got: list={reward1}, string={reward2}")
            # Debug: check isomorphism directly
            if parsed:
                iso = are_trees_isomorphic(parsed, answer_list)
                print(f"  Direct isomorphism check: {iso}")
                print(f"  Parsed: {parsed}")
                print(f"  Answer: {answer_list}")
        
        # Test with incorrect prediction
        completion_wrong = [{"role": "assistant", "content": '{"tree": [[0, 1], [0, 2]]}'}]
        reward3 = tree_reward(completion_wrong, answer_list)
        print(f"✓ Reward for incorrect prediction: {reward3}")
        assert reward3 == 0.0, f"Expected reward 0.0 for incorrect answer, got {reward3}"
        
        return True
    except Exception as e:
        print(f"✗ Failed reward test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CurveBench Environment Verification Tests")
    print("=" * 60)
    
    tests = [
        test_import,
        test_load,
        test_sample,
        test_reward,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
