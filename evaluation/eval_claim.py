from sklearn.metrics import precision_recall_fscore_support


def evaluate_claim_verification(predictions, ground_truth, labels = ["Supported", "Refuted"]):
    """
    """

    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=labels,
        average='macro',
        zero_division=0  # Avoid division by zero if a class is missing
    )

    return {
        "precision": f"{precision * 100:.1f}",
        "recall": f"{recall * 100:.1f}",
        "macro_f1": f"{f1 * 100:.1f}"
    }
    
    

    
