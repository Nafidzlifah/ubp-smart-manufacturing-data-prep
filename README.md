# UBP Smart Manufacturing Dataset (Synthetic)

Purpose: teaching dataset for IF1220038 Pembelajaran Mendalam, especially data preparation, representation, MLP/CNN/RNN-LSTM, training/evaluation, diagnosis, interpretation, and integrative mini-project.

## Files
- production_tabular_raw.csv — 6,000 production-cycle records with controlled missing values, outliers, rare category, grouping and class imbalance.
- production_tabular_reference_clean.csv — reference-clean version for instructor validation; not intended as the first student input.
- sensor_timeseries_raw.csv — 8,640 ordered sensor records across 12 machines.
- visual_inspection/ — 300 synthetic 64x64 grayscale inspection images (240 Good, 60 Defect).
- data_dictionary.xlsx — variable definitions and pedagogical purpose.
- INSTRUCTOR_ground_truth.csv — instructor-only design notes; do not distribute initially to students.

## Important
All records and images are fully synthetic. They do not describe a real factory, employee, machine, or production process. Values are plausible for educational use but must not be treated as engineering specifications.

## Suggested OBE use
Sub-CPMK 11.2.1: raw tabular inspection, missing values, encoding, scaling, split, leakage control, tensor/batch preparation.
Sub-CPMK 11.2.2: MLP/CNN/RNN-LSTM applications using the corresponding modalities.
Sub-CPMK 11.2.3: training-validation-testing and parameter experiments.
Sub-CPMK 11.3.1: performance comparison, learning curves, over/underfitting and error analysis.
Sub-CPMK 11.3.2: interpretation, limitations, communication, and integrative solution.

Random seed for reproducible fabrication: 2022.
