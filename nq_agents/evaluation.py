from collections import defaultdict
from typing import List, Dict, Any
from natural_questions import eval_utils
from natural_questions import nq_eval
import gzip

def evaluate_predictions(gold_path, predictions_path) -> Dict[str, float]:
    """
    Evaluate predictions against gold examples.
    
    Args:
        gold_examples: List of annotated examples with gold answers
        predictions: List of predicted answer strings
    
    Returns:
        Dictionary containing evaluation metrics
    """
    temp_gold_path = f"{gold_path}.gz"
    with open(gold_path, 'rb') as f_in:
        with gzip.open(temp_gold_path, 'wb') as f_out:
            f_out.writelines(f_in)

    nq_gold_dict = eval_utils.read_annotation(
        temp_gold_path, n_threads=10
    )
    nq_pred_dict = eval_utils.read_prediction_json(predictions_path)
    long_answer_stats, short_answer_stats = nq_eval.score_answers(nq_gold_dict,
                                                        nq_pred_dict)
    print(f"Short answer stats: {short_answer_stats}")
    
    return short_answer_stats