# Evaluation Summary {YYYY-MM-DD}

## Model Information
- Model checkpoint: {Checkpoint path}
- Training date: {Date model was trained}
- Parameters: {Number} million parameters

## Test Dataset
- Test set size: {Number} samples
- Categories distribution:
  - {Category 1}: {Percentage}%
  - {Category 2}: {Percentage}%
  - {Category N}: {Percentage}%

## Classification Metrics
| Metric    | Overall | {Class 1} | {Class 2} | {Class N} |
|-----------|---------|-----------|-----------|-----------|
| Accuracy  | {Number}| -         | -         | -         |
| Precision | {Number}| {Number}  | {Number}  | {Number}  |
| Recall    | {Number}| {Number}  | {Number}  | {Number}  |
| F1 Score  | {Number}| {Number}  | {Number}  | {Number}  |

## Confusion Matrix
| Predicted → <br> Actual ↓ | {Class 1} | {Class 2} | {Class N} |
|---------------------------|-----------|-----------|-----------|
| {Class 1}                 | {Number}  | {Number}  | {Number}  |
| {Class 2}                 | {Number}  | {Number}  | {Number}  |
| {Class N}                 | {Number}  | {Number}  | {Number}  |

## Error Analysis
- Most common false positives: {Description}
- Most common false negatives: {Description}
- Edge cases identified: {Description}

## Text Generation Quality
- Average perplexity: {Number}
- BLEU score (if applicable): {Number}
- Human evaluation rating (if performed): {Number}/10

## Inference Performance
- Average latency: {Number} ms
- Throughput: {Number} requests/second
- Memory usage: {Number} MB

## Recommendations
- {Specific recommendations for improving model performance}
- {Suggestions for addressing identified issues}
- {Next steps for model development}

## Generated by
Agent: Evaluator
Date: {YYYY-MM-DD HH:MM}