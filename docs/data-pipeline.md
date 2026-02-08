# Data Pipeline

How TrustECG loads, processes, and prepares ECG data for training and inference.

---

## PTB-XL Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/) is the largest publicly available clinical ECG dataset.

| Property       | Value                           |
| -------------- | ------------------------------- |
| Total records  | 21,801                          |
| Format         | 12-lead ECG                     |
| Duration       | 10 seconds per recording        |
| Sampling rates | 100 Hz and 500 Hz               |
| Annotations    | Cardiologist-verified SCP codes |
| Source         | University Hospital, Germany    |

### Metadata Files

**`ptbxl_database.csv`** contains one row per ECG:

| Column        | Description                                   |
| ------------- | --------------------------------------------- |
| `ecg_id`      | Unique record identifier (1-21799)            |
| `filename_lr` | Path to 100 Hz recording                      |
| `filename_hr` | Path to 500 Hz recording                      |
| `scp_codes`   | Dict of SCP diagnostic codes with likelihoods |
| `strat_fold`  | Pre-assigned stratification fold (1-10)       |
| `age`, `sex`  | Patient demographics                          |

**`scp_statements.csv`** maps SCP codes to diagnostic superclasses:

```
SCP Code → diagnostic_class → Superclass (NORM, MI, STTC, CD, HYP)
```

### Label Extraction

We aggregate SCP codes into 5 binary superclass labels:

```python
import ast
import pandas as pd

# Load metadata
Y = pd.read_csv("dataset/ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load SCP → superclass mapping
agg_df = pd.read_csv("dataset/scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Map each record to its superclasses
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)
```

Result: each record gets a list like `["NORM"]` or `["MI", "STTC"]`.

### Class Distribution

| Class | Count | Prevalence | Description            |
| ----- | ----- | ---------- | ---------------------- |
| NORM  | 9,438 | 43.3%      | Normal                 |
| MI    | 4,134 | 19.0%      | Myocardial Infarction  |
| STTC  | 5,078 | 23.3%      | ST/T Change            |
| CD    | 4,891 | 22.5%      | Conduction Disturbance |
| HYP   | 2,258 | 10.4%      | Hypertrophy            |

Note: percentages sum to >100% because records can have multiple labels (~27% are multi-label).

<p align="center">
  <img src="../figures/01_class_distribution.png" alt="Class Distribution" width="600"/>
</p>

<p align="center">
  <img src="../figures/02_co_occurrence.png" alt="Co-occurrence" width="500"/>
</p>

---

## Train / Validation / Test Split

We use PTB-XL's pre-defined stratification folds to prevent patient leakage:

| Split      | Folds | Records | Percentage |
| ---------- | ----- | ------- | ---------- |
| Train      | 1-8   | 17,441  | ~80%       |
| Validation | 9     | 2,203   | ~10%       |
| Test       | 10    | 2,157   | ~10%       |

```python
train_mask = Y.strat_fold.isin(range(1, 9))
val_mask   = Y.strat_fold == 9
test_mask  = Y.strat_fold == 10
```

This ensures:

- No patient appears in multiple splits
- Class distribution is preserved across splits
- Results are reproducible (same split as the original PTB-XL paper)

---

## Signal Loading

ECG signals are stored in [WFDB format](https://physionet.org/content/wfdb-python/) (`.hea` header + `.dat` data).

```python
import wfdb

# Load a single record (100 Hz)
record = wfdb.rdrecord("dataset/records100/00000/00009_lr")

signal = record.p_signal  # Shape: (1000, 12) — 10 sec × 12 leads
signal = signal.T          # Transpose to (12, 1000) — leads × time
```

**File naming convention:**

```
dataset/records100/{folder:05d}/{ecg_id:05d}_lr.hea
```

Where `folder = (ecg_id - 1) // 1000 * 1000`

Example: ECG ID `9` → `dataset/records100/00000/00009_lr`

---

## Preprocessing

The `ECGPreprocessor` class applies two steps:

### Step 1: Bandpass Filter (0.5–40 Hz)

Removes noise outside the clinically relevant frequency range:

- **Below 0.5 Hz**: Baseline wander from breathing and movement
- **Above 40 Hz**: High-frequency noise, muscle artifacts, power line interference

Implementation uses a 2nd-order Butterworth filter with zero-phase filtering (`scipy.signal.filtfilt`):

```python
from scipy.signal import butter, filtfilt

nyquist = 0.5 * 100  # 50 Hz for 100 Hz sampling rate
b, a = butter(N=2, Wn=[0.5/nyquist, 40.0/nyquist], btype="band")
filtered = filtfilt(b, a, signal_lead)
```

### Step 2: Z-Score Normalization

Standardizes the signal to zero mean and unit variance:

$$x_{\text{norm}} = \frac{x - \mu}{\sigma + \epsilon}$$

Where $\epsilon = 10^{-8}$ prevents division by zero. Normalization is applied **globally** across all 12 leads (not per-lead), preserving relative amplitude differences between leads.

### Complete Preprocessor

```python
class ECGPreprocessor:
    def __init__(self, lowcut=0.5, highcut=40.0, fs=100):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs

    def __call__(self, signal):
        # signal shape: (12, 1000)
        nyquist = 0.5 * self.fs
        b, a = butter(N=2, Wn=[self.lowcut/nyquist, self.highcut/nyquist], btype="band")
        filtered = np.array([filtfilt(b, a, signal[i]) for i in range(signal.shape[0])])
        mean = filtered.mean()
        std = filtered.std() + 1e-8
        return ((filtered - mean) / std).astype(np.float32)
```

**Important**: The same preprocessing must be applied during both training and inference. A mismatch here will cause incorrect predictions (we learned this the hard way).

---

## Data Flow Summary

```
PTB-XL CSV ──→ SCP code parsing ──→ 5 binary labels per record
     │
     └──→ wfdb.rdrecord() ──→ Raw signal (12, 1000)
                                    │
                                    ▼
                            Bandpass filter (0.5-40 Hz)
                                    │
                                    ▼
                            Z-score normalization
                                    │
                                    ▼
                         Preprocessed tensor (12, 1000) float32
                                    │
                                    ▼
                           Model input (batch, 12, 1000)
```

---

**Next**: [Model Architecture](architecture.md) to understand how the neural network processes these signals.
