from collections import defaultdict
from typing import List, Dict, Any

def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    return answer.lower().strip()

def evaluate_predictions(gold_examples: List[Dict], predictions: List[str]) -> Dict[str, float]:
    """
    Evaluate predictions against gold examples.
    
    Args:
        gold_examples: List of annotated examples with gold answers
        predictions: List of predicted answer strings
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if len(gold_examples) != len(predictions):
        raise ValueError("Number of predictions doesn't match number of examples")
    
    metrics = defaultdict(int)
    total = len(gold_examples)
    
    for example, pred in zip(gold_examples, predictions):
        # Get gold answers from annotations
        gold_answers = []
        for annotation in example['annotations']:
            if annotation['short_answers']:
                for short_answer in annotation['short_answers']:
                    answer_tokens = example['document_tokens'][short_answer['start_token']:short_answer['end_token']]
                    gold_answers.append(' '.join(answer_tokens))
        
        # Normalize answers
        pred = normalize_answer(pred)
        gold_answers = [normalize_answer(ans) for ans in gold_answers]
        
        # Check exact match
        if pred in gold_answers:
            metrics['exact_match'] += 1
            
        # Check partial match (if pred is substring of any gold answer or vice versa)
        if any(pred in ans or ans in pred for ans in gold_answers):
            metrics['partial_match'] += 1
    
    # Calculate percentages
    metrics['exact_match'] = (metrics['exact_match'] / total) * 100
    metrics['partial_match'] = (metrics['partial_match'] / total) * 100
    
    return dict(metrics) 