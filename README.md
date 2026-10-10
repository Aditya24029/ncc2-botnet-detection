# Botnet Detection on NCC-2 with Tabular Transformers

Research internship project, Jawaharlal Nehru University (JNU), under Prof. Satish Chand.

Binary botnet-vs-normal classification on network flows from the **NCC-2 Simultaneous Botnet Dataset**,
which has three sensors capturing the same network. The project compares three transformer
architectures for tabular data:

| Model | Setup |
|---|---|
| **FT-Transformer** | Trained on Sensor 1, then fine-tuned on Sensors 2 and 3 (transfer across sensors) |
| **TabTransformer** | Same transfer setup as FT-Transformer |
| **SAINT** | Implemented from scratch in PyTorch, including **intersample attention** (each row attends to the other rows in the batch), trained per sensor |

## Pipeline

```
raw .binetflow  ->  01 preprocessing  ->  processed CSVs
                                          |
           +------------------------------+------------------------------+
           |                              |                              |
   02-04 FT-Transformer           05-07 TabTransformer             08-10 SAINT
   (S1 pretrain -> S2, S3)        (S1 pretrain -> S2, S3)          (per sensor)
           |                              |                              |
           +------------------> results/*.json <-------------------------+
                                          |
                               11 results comparison
```

**Evaluation protocol.** Every model uses the same stratified 70/15/15 split (seed 42), so all
models are scored on identical test rows. Validation selects the checkpoint, and the test set is
used once. The data is heavily imbalanced (about 3-7% botnet flows), so **macro F1 and botnet-class
F1 are the headline metrics**, not accuracy.

## Results

All models scored on the same held-out test set (15% of each sensor, stratified).

| Model | Sensor | Accuracy | Botnet precision | Botnet recall | Botnet F1 | Macro F1 |
|---|---|---|---|---|---|---|
| FT-Transformer | 1 | 0.9968 | 0.9638 | 0.9279 | **0.9455** | **0.9719** |
| TabTransformer | 1 | 0.9962 | 0.9436 | 0.9291 | 0.9363 | 0.9672 |
| SAINT | 1 | 0.9360 | 0.3115 | 0.9467 | 0.4688 | 0.7174 |
| FT-Transformer | 2 | 0.9924 | 0.9495 | 0.9241 | **0.9366** | **0.9663** |
| TabTransformer | 2 | 0.9899 | 0.9352 | 0.8954 | 0.9149 | 0.9547 |
| SAINT | 2 | 0.9388 | 0.4977 | 0.9197 | 0.6458 | 0.8062 |
| FT-Transformer | 3 | 0.9910* | 0.9592* | 0.9201* | 0.9392* | 0.9672* |
| TabTransformer | 3 | 0.9866 | 0.9245 | 0.8967 | **0.9104** | **0.9516** |
| SAINT | 3 | 0.9292 | 0.5176 | 0.9500 | 0.6701 | 0.8152 |

**Takeaways**

- **FT-Transformer is the strongest overall**, with the best botnet F1 and macro F1 on every sensor where it has run.
- **Transfer across sensors works.** Models pretrained on Sensor 1 and fine-tuned on Sensors 2 and 3 keep botnet F1 above 0.91.
- **SAINT trades precision for recall.** With class-weighted loss it catches 92-95% of botnet flows (the highest recall here), but precision drops to 31-52%. In intrusion detection a missed bot costs more than a false alarm, so this is a deliberate operating point. Tuning the decision threshold on validation is the next step.

## Repository layout

```
notebooks/   01-11, run in order
src/         shared config, split and metrics (common.py)
results/     one JSON per model x sensor
data/        (not committed) raw/ and processed/
models/      (not committed) saved checkpoints
```

## Reproduce

1. Download NCC-2 from the official source: https://data.mendeley.com/datasets/8dpt85jrhp/2 . Place it as
   `data/raw/sensor{1,2,3}/sensor{1,2,3}.binetflow`.
2. `pip install -r requirements.txt`
3. Run notebooks 01 to 11 in order from inside `notebooks/`.

Tested on Windows with an NVIDIA RTX 3050 (6 GB).

## Limitations

- `Hour`/`Minute`/`Second` are derived from the capture timestamp. Because botnet activity in
  NCC-2 occurs in specific time windows, these features may let models learn *when* rather than
  *what*. An ablation without them is future work.
- Ports are ordinal-encoded as categoricals, and SAINT treats all features as continuous tokens.

## Acknowledgements

Project idea by Ms. Anjali (PhD scholar, JNU). Supervised by Prof. Satish Chand.
