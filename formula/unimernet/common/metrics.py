from unimernet.datasets.datasets.bleu import compute_bleu
from unimernet.common.tokenizer_13a import Tokenizer13a


def compute_corpus_bleu(predictions, references, max_order=4):
    if not predictions:
        return 0.0

    tokenizer = Tokenizer13a()
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    tokenized_predictions = [tokenizer(prediction) for prediction in predictions]
    tokenized_references = [
        [tokenizer(reference) for reference in reference_group]
        for reference_group in references
    ]

    bleu, _, _, _, _, _ = compute_bleu(
        tokenized_references,
        tokenized_predictions,
        max_order=max_order,
        smooth=False,
    )
    return bleu
