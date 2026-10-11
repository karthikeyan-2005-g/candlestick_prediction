# Dataset, training, evaluation, and inference

## Supplied archive and label caveat

The supplied folders are `archive/Train/Down`, `archive/Train/Up`,
`archive/Test/Down`, and `archive/Test/Up`. The loader maps the folder labels
to `Down=0` and `Up=1`. The archive has no source README, label-generation
rule, timestamps, or split methodology. Therefore `Down` and `Up` are treated
as dataset classes only; they are not asserted to mean the future candle
direction.

Before training, inspect image counts and verify image decoding. Evaluation
checks exact file hashes and matching symbol/index filename stems. The
observed shared stems should be interpreted as a warning, not proof of leakage;
without timestamps/source information the supplied Test partition cannot be
certified as an independent chronological holdout.

## Input preprocessing and model

Images are decoded as RGB, resized by OpenCV with `INTER_AREA` to 128 × 128,
converted to `float32`, and scaled from `[0,255]` to `[0,1]`. Training and
inference call the same array preprocessing function. The tensor layout is
`[3,128,128]`; batching yields `[N,3,128,128]`. The CNN emits two logits in
`Down`, `Up` order.

Checkpoints store the class map, class names, preprocessing configuration,
image size, model version, and training/validation metadata. Inference rejects
missing or incompatible weights; it never initializes random weights as a
fallback.

## Training and validation

Training reads only `archive/Train`. Because the archive has no source metadata
or stable sample IDs, validation groups images by the `QQQ` or `SPY` filename
prefix, treating it as an instrument-level proxy. This is an explicit
assumption, not verified dataset provenance; the numeric suffix is not treated
as a unique sample ID. One whole prefix is held out, selected deterministically
to be closest to the requested validation fraction while preserving both
classes in each partition. With only two groups, the default 20% fraction
cannot be reached: QQQ contains 504 of 1,433 Train images (35.2%), so QQQ is
held out and SPY is used for training. The achieved fraction and group names
are saved in checkpoint metadata. `archive/Test` is not read by the trainer
and is never used for split selection or model training. This instrument-level
holdout does not verify timestamp or window independence; use source metadata
with timestamps and stable observation IDs for a defensible temporal split.

From Windows Command Prompt, run at the project root:

```bat
python -m ml.train --data-dir archive --train-split Train --output models\candlestick_cnn.pth --epochs 30 --batch-size 32 --learning-rate 0.001 --validation-fraction 0.2
```

The trainer reports class counts and training/validation loss and accuracy per
epoch. It uses cross-entropy, AdamW, deterministic split seeding, and early
stopping; it saves the best validation-accuracy checkpoint (validation loss
breaks ties) to `models/candlestick_cnn.pth`, with `training_metrics.json`
beside it.

## Final evaluation

Only after training is complete, run:

```bat
python -m ml.evaluate --data-dir archive --train-split Train --test-split Test --checkpoint models\candlestick_cnn.pth --output models\evaluation_metrics.json
```

Evaluation reports accuracy, per-class precision/recall/F1, confusion matrix,
training-majority baseline, and an overlap audit. Metrics measure agreement
with the supplied Test folder labels. Because the source label rule and split
provenance are undocumented and filenames overlap between the supplied splits,
these are not certifiable independent chronological or future-market metrics.
Do not tune hyperparameters using Test results.

## Optional OHLCV-derived data

For labels with a verifiable financial meaning, obtain properly licensed,
timestamped OHLCV history and use `scripts/prepare_dataset.py`. That separate
pipeline labels the next candle by close versus open, excludes doji bars, and
splits chronological raw data before generating overlapping windows. It is
separate from and does not rewrite `archive/`.

## Prediction confidence and limitations

The prediction service applies softmax and returns the winning archive class,
`Down` or `Up`, with its confidence score. A low score is flagged without
changing the predicted dataset class. Softmax scores are not necessarily
calibrated probabilities. Image classification of an undocumented dataset
does not establish future market direction, trading performance, or financial
advice.
