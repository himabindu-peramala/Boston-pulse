# Boston Pulse — Model Development & Evaluation

## Results Summary
| Model | Retrieval Accuracy |
|---|---|
| all-MiniLM-L6-v2 | 71.4% |
| all-mpnet-base-v2 | 69.0% |
| **BAAI/bge-base-en-v1.5** | **76.2%** ✅ |

## How to Run
cd backend
python -m evaluation.eval_runner
python -m evaluation.bias_detector
python -m evaluation.run_all_experiments
mlflow ui
