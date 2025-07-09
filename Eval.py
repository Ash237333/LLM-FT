import torch
from sklearn.metrics import recall_score

P_TOKEN = 11
U_TOKEN = 12
ALPHA = 0.088

def compute_metrics(eval_pred, predict_threshold=True):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Predicted class for each example 1=synthesizable, 0=impossible
    logits_at_pos = logits[:, -2, [P_TOKEN, U_TOKEN]]
    probs = torch.softmax(logits_at_pos, dim=-1)
    p_probs = probs[:, 0]

    # Converts labels to the same binary format
    true_at_pos = labels[:, -2]
    true_class = (true_at_pos == P_TOKEN).int()

    # Creates masks to differentiate known positives from unlabeled
    known_positives_mask = (true_at_pos == P_TOKEN)
    unlabeled_mask = (true_at_pos == U_TOKEN)

    # Finds positive probability for all unlabelled examples
    if predict_threshold:
        unlabeled_p_probs = p_probs[unlabeled_mask]

        # Calculates threshold for classification to line up with alpha expectation
        k = int(ALPHA * len(unlabeled_p_probs))

        # Handles edge-case to prevent errors
        if k > 0:
            threshold = torch.topk(unlabeled_p_probs, k).values.min().item()
        else:
            threshold = 1.0
    else:
        threshold = 0.5

    # Calculates the predicted class for each example
    predicted_class = (p_probs > threshold).int()

    # Recall (TPR) calculated on True Positives only
    true_pos = true_class[known_positives_mask]
    pred_pos = predicted_class[known_positives_mask]
    recall = recall_score(true_pos.cpu().numpy(), pred_pos.cpu().numpy())

    # Calculate predicted positives in the unlabelled set

    num_pred_positives_in_unlabeled = predicted_class[unlabeled_mask].sum().item()

    # Calculate an estimate of how many of those positives are actually positive
    num_TP = pred_pos.sum().item()
    num_FP = num_pred_positives_in_unlabeled * (1 - ALPHA)

    # Use previous two steps to estimate precision
    # Handle div 0 edge-case to prevent error
    if (num_TP + num_FP) > 0:
        precision_estimate = num_TP / (num_TP + num_FP)
    else:
        precision_estimate = 0.0

    return {
        "precision": precision_estimate,
        "recall": recall,
        "threshold": threshold
    }
