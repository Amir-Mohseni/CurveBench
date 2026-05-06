#!/bin/bash
# Test pinference custom model with image + enable_multimodal
# Run from curvebench_env: ./test_pinference_image.sh

set -e
if [ -f configs/lab/.env ]; then
  set -a
  source configs/lab/.env
  set +a
fi

if [ -z "$PRIME_API_KEY" ]; then
  echo "Error: PRIME_API_KEY not set. Add it to configs/lab/.env or export it."
  exit 1
fi

# 1x1 pixel PNG (minimal valid image)
IMG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

echo "Testing pinference with custom model + image + enable_multimodal..."
curl -s -X POST https://api.pinference.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PRIME_API_KEY" \
  -d "{
    \"model\": \"Qwen/Qwen3-VL-4B-Instruct:oolc6a87xnbmy1wyuek71ieq\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"What color is this image? Reply in one word.\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
        ]
      }
    ],
    \"max_tokens\": 50
  }" | python3 -m json.tool 2>/dev/null || cat
