# Dataset

This project uses the **Mental Health Counseling Conversations** dataset by Amod Sahasrabude.

- **Source**: [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) on Hugging Face
- **Format**: JSONL (one JSON object per line)
- **Size**: 3,512 question-answer pairs across 995 unique questions
- **Fields**: `Context` (user question) and `Response` (counselor answer)
- **License**: RAIL-D (see `LICENSE-RAIL-D.txt` in this directory)

## Download

Download the dataset and place `combined_dataset.json` in this directory:

```bash
# Option 1: Using the Hugging Face CLI
huggingface-cli download Amod/mental_health_counseling_conversations --local-dir data/

# Option 2: Manual download
# Visit https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
# and download combined_dataset.json into this directory.
```

## License Notice

The dataset is released under the Responsible AI License (RAIL-D). Non-commercial research
use is royalty-free. Commercial use requires a mandatory donation. See `LICENSE-RAIL-D.txt`
for full terms.
