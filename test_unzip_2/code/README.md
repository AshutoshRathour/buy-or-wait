# HackerRank Orchestrate: Buy or Wait? Solution

## Setup Instructions

1. Ensure you have Python 3.9+ installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requirements are mostly standard (e.g., pandas).*

## Run Instructions

To generate `output.csv` predictions on the dataset:

```bash
python main.py --data-dir ../dataset
```

This will run the deterministic financial engine across all files in `../dataset/` and write `output.csv` to the parent directory.

To run the evaluation validation logic:
```bash
python evaluation/main.py --data-dir ../dataset
```
