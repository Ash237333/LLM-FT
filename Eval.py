import torch
from sklearn.metrics import recall_score

P_TOKEN = 47
U_TOKEN = 52
ALPHA = 0.088


def compute_metrics(eval_pred, predict_threshold=True):
    """
        Computes precision, recall, and decision threshold based on model predictions and true labels.

        This function is designed for binary classification with two special tokens representing
        positive ("P_TOKEN") and unlabeled ("U_TOKEN") classes. It extracts logits at a specific
        position in the sequence, applies softmax to obtain probabilities, and calculates metrics.

        If `predict_threshold` is True, the function dynamically determines a classification threshold
        based on the top ALPHA fraction of predicted positive probabilities within the unlabeled set.
        Otherwise, it uses a fixed threshold of 0.5.

        Args:
            eval_pred (EvalPrediction): An object with two attributes:
                - predictions (np.ndarray or torch.Tensor): Logits output by the model.
                - label_ids (np.ndarray or torch.Tensor): True labels corresponding to the inputs.
            predict_threshold (bool, optional): Whether to compute a dynamic threshold based on the
                unlabeled samples' predicted probabilities. Defaults to True.

        Returns:
            dict: A dictionary with the following keys:
                - "precision" (float): Estimated precision of predicted positives considering ALPHA.
                - "recall" (float): Recall score computed on known positive samples.
                - "threshold" (float): The probability threshold used to classify positives.
        """
    logits = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids)

    logits_at_pos = logits[:, -2, [P_TOKEN, U_TOKEN]]
    probs = torch.softmax(logits_at_pos, dim=-1)
    p_probs = probs[:, 0]

    true_at_pos = labels[:, -2]
    true_class = (true_at_pos == P_TOKEN).int()

    known_positives_mask = (true_at_pos == P_TOKEN)
    unlabeled_mask = (true_at_pos == U_TOKEN)

    if predict_threshold:
        unlabeled_p_probs = p_probs[unlabeled_mask]

        k = int(ALPHA * len(unlabeled_p_probs))

        if k > 0:
            threshold = torch.topk(unlabeled_p_probs, k).values.min().item()
        else:
            threshold = 1.0
    else:
        threshold = 0.5

    predicted_class = (p_probs > threshold).int()

    true_pos = true_class[known_positives_mask]
    pred_pos = predicted_class[known_positives_mask]
    recall = recall_score(true_pos.cpu().numpy(), pred_pos.cpu().numpy())

    num_pred_positives_in_unlabeled = predicted_class[unlabeled_mask].sum().item()

    num_TP = pred_pos.sum().item()
    num_FP = num_pred_positives_in_unlabeled * (1 - ALPHA)

    if (num_TP + num_FP) > 0:
        precision_estimate = num_TP / (num_TP + num_FP)
    else:
        precision_estimate = 0.0

    return {
        "precision": precision_estimate,
        "recall": recall,
        "threshold": threshold
    }
