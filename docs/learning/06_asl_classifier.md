# 06 — The ASL Classifier

## 1. What it does

Takes one 63-value normalized feature vector (Phase 5) and predicts
which ASL letter it most likely represents, plus a confidence score.
See [asl_recognizer.py](../../src/vision/asl_recognizer.py) and the
training pipeline in [scripts/train_asl.py](../../scripts/train_asl.py).

## 2. Supervised learning, in this project's terms

**Supervised learning** = learning from labeled examples: here, pairs
of `(63 numbers, a letter)`. **Features** are the 63 numbers (the
input); the **label** is the letter (the answer we want to predict).
Training a classifier means: show it many `(features, label)` pairs,
and it learns a rule that maps new, unseen feature vectors to a
predicted label.

## 3. The dataset — honestly

There is **no downloaded ASL landmark dataset** in this project. A
raw-image ASL alphabet dataset exists publicly, but a *landmark-level*
one (already-extracted 21-point coordinates, matching exactly how this
project represents a hand) isn't something we could find and verify
the license/provenance of — and fabricating or silently substituting
one would violate this project's core rule: **never fabricate a
dataset.**

Instead: [scripts/collect_data.py](../../scripts/collect_data.py) lets
you record your own samples with your own webcam, each one MediaPipe
landmarks + Phase 5's normalization + a label. This is honest by
construction — every row in `data/asl_landmarks.csv` is a real
detection MediaPipe made on a real, live camera frame; the label is
literally what letter you told the script you were signing at the
time. Format: 63 feature columns (`f0`...`f62`) + one `label` column,
one row per captured sample.

**Trade-off you should expect**: a small, self-collected dataset
(one signer, one lighting setup, one camera) will not be as robust as
a large public dataset with hundreds of signers. Recognition accuracy
and how well it tolerates an imperfect hand position depend directly
on **how many varied samples you record per letter** — more samples,
and more variation between them (angle, distance, position in frame)
per letter, is what actually buys robustness, not any code trick.

## 4. Random Forest, and why not a CNN

A **Random Forest** is an ensemble of many decision trees, each
trained on a slightly different random subset of the data/features;
its final prediction is the majority vote across all the trees. It's
the right tool here specifically *because* the input is already a
small, clean, structured 63-number vector (not raw pixels) — for that
kind of tabular input, a Random Forest is fast to train even on a
small dataset, doesn't need a GPU, resists overfitting to noise better
than a single decision tree, and — importantly for a viva — is
genuinely explainable (you can inspect which of the 63 features each
tree actually splits on). A CNN is built for raw-image inputs where
Phase 5's whole point (a small, structured, normalized representation)
would be thrown away; the project's spec deliberately says "don't
build a CNN unless absolutely necessary," and here it isn't.

## 5. The training pipeline

```
data/asl_landmarks.csv
 -> load_dataset(): features (X) + labels (y)
 -> train_test_split(): 80% train, 20% held-out test
 -> RandomForestClassifier(n_estimators=150).fit(X_train, y_train)
 -> evaluate on X_test (never seen during training)
 -> print REAL accuracy + classification_report
 -> joblib.dump(...) -> models/asl_classifier.pkl
```

**Why a held-out test set matters**: evaluating on the same data the
model trained on would let it "cheat" by memorizing rather than
generalizing — the reported accuracy would be meaninglessly high. The
20% test split never influences training, so its accuracy is an
honest estimate of how the model does on hand positions it hasn't
seen before.

## 6. Probability, confidence, and overfitting

- **Probability/confidence**: a Random Forest's `predict_proba()`
  returns, for one input, the fraction of its trees that voted for
  each class. `ASLRecognizer.predict()` takes the top class and
  reports that fraction as "confidence" — literally "how much of the
  forest agreed."
- **Overfitting** = a model that memorized the training examples'
  quirks instead of learning the letter's actual shape, so it does
  great on training data and poorly on new data. Symptom: near-100%
  training accuracy but much lower test accuracy. With a small,
  self-collected dataset this is a real risk — it's exactly why
  `scripts/train_asl.py` reports **test** accuracy, never training
  accuracy, as "the" number.
- **This project never fabricates accuracy.** Whatever
  `scripts/train_asl.py` prints after you run it on your own data is
  the real, measured number — not a claim made in this document.

## 7. Why landmark classification suits this problem

Covered in depth in `docs/learning/04_hand_landmarks.md` §5 — the
short version: 63 clean, normalized numbers is a far easier thing for
a small classifier to learn from than raw pixels, and normalization
(Phase 5) already removed position/scale as confounding variables, so
the Random Forest only has to learn actual hand *shape* differences
between letters.

## 8. The J/Z limitation

`config.ASL_LABELS` deliberately excludes J and Z. Both are "drawn" in
ASL — the hand traces a shape in the air over time — which a
single-frame landmark snapshot fundamentally cannot capture (there's
no "one static position" for a moving letter). Recognizing them would
need a sequence model (e.g. tracking landmarks across N frames), which
is out of scope here. This is a documented limitation, not a bug —
see the spec's honesty requirement on ASL scope.

## 9. Input / Output

- Input: one 63-value feature vector (Phase 5's output).
- Output: `(letter, confidence)` — a single predicted character
  0-9-2 4 letters from `ASL_LABELS`) plus a float in `[0, 1]`. Returns
  `(None, 0.0)` if no model is loaded — never fabricates a guess.

## 10. Connection to other components

Takes feature vectors from `landmark_utils.py` (Phase 5). Feeds
predictions into `stabilizer.py` (Phase 7), which decides whether a
prediction is stable enough to actually accept. Contains zero Wordle
knowledge — same boundary rule as every other vision-layer file.

## 11. Common errors

- **Training with too few samples per letter** — `train_asl.py` warns
  if any letter has fewer than 5 samples; expect that letter's
  precision/recall to be unreliable.
- **Forgetting to restart the Streamlit app after training** — the
  model is loaded once into `st.session_state` when the app starts;
  a newly-trained `.pkl` file isn't picked up until you restart.
- **Testing on the same data you trained on** — always check the
  reported number is the *test*-set accuracy, not training accuracy.

## 12. Likely viva questions

- **"Why Random Forest instead of a neural network?"** — The input is
  already a small, structured, normalized 63-number vector, not raw
  pixels; a Random Forest is fast, needs little data, resists
  overfitting reasonably well, and is explainable — a CNN would be
  solving a harder problem (learn features from pixels) that Phase 5
  already solved.
- **"Where did your training data come from?"** — Self-recorded with
  `scripts/collect_data.py`, real MediaPipe detections on a real
  webcam — not downloaded, not fabricated.
- **"What's your model's actual accuracy?"** — Whatever
  `scripts/train_asl.py` printed on the held-out test set the last
  time it was run — a real, measured number, never assumed.
- **"Why can't it recognize J or Z?"** — They're dynamic (motion-based)
  letters; a single-frame landmark classifier has no concept of
  motion over time.
- **"What does 'confidence' mean here?"** — The fraction of the Random
  Forest's individual trees that voted for the predicted letter.
