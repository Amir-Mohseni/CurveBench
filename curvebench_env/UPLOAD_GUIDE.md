# Upload Guide for CurveBench Environment

## Prerequisites ✅

You've already completed:
- ✅ Environment code (`curvebench_env.py`)
- ✅ Dataset on HuggingFace ([AmirMohseni/CurveBench-Easy](https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy))
- ✅ SingleTurnEnv with Rubric implementation
- ✅ Image handling (data URLs)
- ✅ Dependencies in `pyproject.toml`
- ✅ Tests passing (`python test_env.py`)

## Step-by-Step Upload Process

### 1. Install Prime CLI

```bash
# Using uv (recommended)
uv pip install prime

# Or using pip
pip install prime

# Or using uv tool (alternative)
uv tool install prime
```

### 2. Login to Prime Intellect

```bash
prime login
```

This will open a browser for authentication.

### 3. (Optional) Set up Prime Lab Workspace

If you want to follow Prime's recommended structure:

```bash
prime lab setup
```

This creates an `environments/` directory and other recommended folders.

### 4. Test the Environment Locally

Before uploading, verify everything works:

```bash
# From the curvebench_env directory
uv pip install -e .
python test_env.py
```

Expected output:
```
✓ All tests passed! Environment is ready to push.
```

### 5. Upload to Environments Hub

From inside the `curvebench_env` directory:

```bash
# Upload to your personal account
prime env push

# Or upload to a team
prime env push --team <team-username>
```

The first push will:
- Package your environment
- Upload to the Environments Hub
- Make it installable as `amirmohseni/curvebench-env`

### 6. Update Version (for future updates)

When making changes:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or 0.2.0 for bigger changes
   ```

2. Push with auto-bump (alternative):
   ```bash
   prime env push --auto-bump
   ```

### 7. Test the Uploaded Environment

Install and test from the Hub:

```bash
# Install from Hub
prime env install amirmohseni/curvebench-env

# Test with Prime's eval
prime eval run amirmohseni/curvebench-env --model gpt-4o-mini --limit 5
```

## Available Dataset Splits

Your environment supports these splits:

### By Difficulty Level
- `level_1_train`, `level_1_validation`, `level_1_test` (easiest)
- `level_2_train`, `level_2_validation`, `level_2_test` (medium)
- `level_3_train`, `level_3_validation`, `level_3_test` (hardest)

### Combined
- `total_train` (210 samples)
- `total_validation` (45 samples)
- `total_test` (45 samples)

To use a specific split:

```python
from curvebench_env import load_environment

# Use a specific split
env = load_environment(split="level_1_train")
```

## What Happens After Upload?

1. **Evaluation**: Anyone can run evals against your environment using any supported model
2. **Training**: Your environment becomes available for RL training on Prime's platform
3. **Discovery**: Others can find and use your environment via the Hub

## Troubleshooting

### "Multiple top-level modules discovered"
Already fixed! We configured `py-modules = ["curvebench_env"]` in `pyproject.toml`.

### "Rubric.__init__() got unexpected keyword"
Already fixed! We use `Rubric([tree_reward])` with the function wrapped in a list.

### SSL Certificate Errors
If you get SSL errors during `prime env push`, try:
```bash
# On macOS, update certificates
/Applications/Python\ 3.*/Install\ Certificates.command
```

### Dataset Access Issues
Your HuggingFace dataset requires accepting terms. Make sure you're logged in:
```bash
huggingface-cli login
```

## Next Steps

1. **Upload now** using the steps above
2. **Run evaluations** with different VLMs
3. **Share results** with the community
4. **Consider RL training** once you have baseline results

## Support

- Prime Intellect Docs: https://docs.primeintellect.ai/
- Verifiers Docs: https://github.com/PrimeIntellect-ai/verifiers
- HuggingFace Dataset: https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy

---

**Ready to upload!** Just run:
```bash
cd /Users/amirreza/CurveBench/curvebench_env
prime env push
```
