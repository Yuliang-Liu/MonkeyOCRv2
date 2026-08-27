# MonkeyOCRv2 Text Detection

The text detection experiments from the MonkeyOCRv2 paper are split into two
sub-directories, one per detection codebase. Both plug the MonkeyOCRv2-AS
visual encoder (ViTAEv2-S, 21M parameters) into an existing detector and
compare it against the original ImageNet backbone and the text-specific oCLIP
backbone.

| Directory                      | Detectors     | Benchmarks                                                         |
| ------------------------------ | ------------- | ------------------------------------------------------------------ |
| [`DPText-DETR/`](DPText-DETR/) | DPText-DETR   | Total-Text, CTW1500, ICDAR19-ArT, Rotated Total-Text, Inverse-Text |
| [`mmocr/`](mmocr/)             | DBNet, PSENet | Total-Text, CTW1500, ICDAR2015                                     |

See each directory's `README.md` for the results tables, installation, dataset
preparation, training, evaluation and checkpoint downloads.
