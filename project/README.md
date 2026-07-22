# Microwave Imaging Framework — Module 1 & Module 2

**Development of a Data Processing and Image Reconstruction Framework Using Standardized Measurement Data**

This repository implements the first two modules of a desktop microwave imaging application:

- **Module 1 — Data Acquisition**: import, validate, and organize MATLAB (`.mat`) and Touchstone (`.s1p/.s2p/.s4p/.s8p`) measurement datasets.
- **Module 2 — Signal Preprocessing**: noise filtering, calibration, normalization, background subtraction, and hybrid artifact suppression, producing a cleaned dataset ready for later time-domain conversion / image reconstruction (not implemented in this stage).

Image reconstruction, tumor detection, and visualization beyond signal plots are intentionally **out of scope** for this stage.

---

## 1. Folder Structure

```
project/
main.py                          # Application entry point (launches the GUI)
requirements.txt

gui/
    main_window.py               # Main window; hosts Module 1 + Module 2 tabs
    upload_page.py                # Module 1 GUI: load button, info panel, status log

data_loader/
    loader.py                     # Unified dispatcher (Module 1 entry point)
    matlab_loader.py              # .mat loader (legacy + v7.3/HDF5)
    touchstone_loader.py          # .sNp loader (via scikit-rf)
    validator.py                  # File & content validation
    dataset_info.py               # MicrowaveDataset / DatasetSummary data model

preprocessing/
    filtering.py                  # Noise filtering (moving avg, Savitzky-Golay, Butterworth)
    calibration.py                # Reference-based & self-calibration
    normalization.py              # Min-max, z-score, max-magnitude normalization
    artifact_suppression.py       # Background subtraction, SVD clutter removal, hybrid method
    preprocessing_pipeline.py     # Pipeline orchestration + signal quality evaluation

utils/
    exceptions.py                 # Custom exception hierarchy
    logger.py                     # Shared logger + GUI status log helper

datasets/
    generate_sample_data.py       # Generates synthetic .mat / .s2p sample files
    (generated sample files)

results/                          # (reserved for future modules' output)

tests/
    test_data_loader.py           # Module 1 unit tests
    test_preprocessing.py         # Module 2 unit tests
```

---

## 2. Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running the Application

```bash
python main.py
```

This opens the desktop GUI with two tabs:

1. **Data Acquisition** — click **Load Dataset**, pick a `.mat` or `.sNp` file, and the Dataset Information panel + Status Log populate automatically. Sample files are provided in `datasets/` (see below).
2. **Signal Preprocessing** — once a dataset is loaded, configure the pipeline (filter/calibration/normalization/artifact-suppression method) and click **Run Preprocessing**. A progress bar tracks pipeline stages; on completion, four signal plots (Original, Processed, Comparison, Frequency Response) and a Processing Summary table (SNR, dynamic range, artifact reduction, etc.) are displayed.

### Generating sample datasets

```bash
python datasets/generate_sample_data.py
```

Produces:
- `sample_matlab.mat` — legacy MAT format, 8 synthetic antenna traces
- `sample_matlab_v73.mat` — MAT v7.3 (HDF5) format
- `sample_touchstone.s2p` — 2-port Touchstone file
- `corrupted.mat` — intentionally invalid file, for exercising error handling

## 4. Running Tests

```bash
python -m unittest discover -s tests -v
```

44 unit tests cover file validation, both MATLAB formats, Touchstone loading, every preprocessing stage individually, and the full pipeline end-to-end (including error paths for corrupted/missing/empty/invalid data).

---

## 5. Module 1 — Data Acquisition, Design Notes

- **`data_loader/loader.py`** is the single entry point (`load_dataset(file_path)`); it dispatches by file extension to `matlab_loader.py` or `touchstone_loader.py` and always returns a canonical `MicrowaveDataset`, so downstream code (Module 2, GUI) never needs to know the original file format.
- **MATLAB loading** supports both legacy (`scipy.io.loadmat`) and MAT v7.3/HDF5 (`h5py`) formats, detected automatically via the file's magic bytes. Because MATLAB exports don't follow one fixed schema, the loader uses name-based and shape-based heuristics to locate the frequency vector and S-parameter array among the file's variables.
- **Touchstone loading** uses `scikit-rf`'s `Network` class, which robustly parses `.s1p/.s2p/.s4p/.s8p` files including comments and port impedances.
- **Validation** (`validator.py`) is applied to every loaded dataset: file existence/extension/size checks, frequency vector sanity (finite, non-negative, monotonic), and S-parameter sanity (correct shape, finite values, non-zero).
- **Dataset Information panel** fields (File Name, File Type, Number of Samples, Number of Frequencies, Frequency Range, Number of Ports, Dataset Size, Available Variables) are produced by `DatasetSummary.to_display_dict()`.

## 6. Module 2 — Signal Preprocessing, Design Notes

Pipeline order (matches the project brief):

```
Raw Dataset → Noise Filtering → Calibration → Normalization →
Background Subtraction → Artifact Suppression → Processed Dataset
```

- **Noise Filtering** (`filtering.py`): moving-average, Savitzky-Golay (polynomial smoothing that preserves resonance peak shape), and Butterworth low-pass — selectable per run.
- **Calibration** (`calibration.py`): reference-based (`S_meas / S_ref`, classic VNA-style normalization) when a reference sweep is supplied, or self-calibration (common-mode offset removal across traces) otherwise.
- **Normalization** (`normalization.py`): min-max, z-score, or max-magnitude — applied to magnitude while preserving phase (phase carries propagation-delay information needed for later time-domain conversion).
- **Background Subtraction / Hybrid Artifact Suppression** (`artifact_suppression.py`): rotation/average-trace subtraction (or explicit background subtraction when a reference is available) followed by SVD-based adaptive clutter removal. The **hybrid** method chains both stages — this two-stage combination follows the confocal microwave imaging literature on skin-artifact removal for antenna-array systems, where a deterministic subtraction stage removes the dominant common-mode reflection and an adaptive/statistical (SVD) stage removes residual correlated clutter that subtraction alone leaves behind.
- **Signal Quality Evaluation** (`preprocessing_pipeline.py`): SNR (dB) estimated from the ratio of smoothed "signal" power to residual "noise" power, dynamic range (dB), magnitude mean/std before & after, and an artifact-reduction ratio based on energy change — all surfaced in the GUI's Processing Summary table.
- The whole pipeline runs on a background `QThread` (`PreprocessingWorker`) so the GUI stays responsive, reporting per-stage progress to drive the progress bar.

## 7. Error Handling

All framework-specific errors derive from `MicrowaveFrameworkError` (`utils/exceptions.py`):

| Exception | Raised when |
|---|---|
| `UnsupportedFileFormatError` | File extension isn't `.mat/.s1p/.s2p/.s4p/.s8p` |
| `CorruptedFileError` | File has a valid extension but can't be parsed |
| `MissingVariableError` | No frequency vector / S-parameter array found in a `.mat` file |
| `EmptyDatasetError` | File is 0 bytes, or parsed data has zero samples |
| `InvalidFrequencyError` | Frequency values are NaN/Inf/negative/non-numeric |
| `InvalidSParameterError` | S-parameter shape mismatch, NaN/Inf values, or all-zero data |
| `DatasetValidationError` | Aggregated validation failure wrapping the above |

Every error surfaces as a clear message in the GUI's Status Log and a `QMessageBox` dialog; the application never crashes on invalid input.

## 8. What's Deliberately Out of Scope Here

Per the project brief, this stage does **not** implement: time-domain conversion, image reconstruction algorithms, tumor/anomaly detection, or 2D/3D spatial visualization. The `processed_dataset.s_parameters` produced by Module 2 is the intended hand-off point for that future work.
