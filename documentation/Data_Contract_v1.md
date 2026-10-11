# Data Contract v1 — UBP Smart Manufacturing (Raw Data)

1. Dataset Name: UBP Smart Manufacturing — Production Tabular
2. Source File: `production_tabular_raw.csv`
3. Sample Definition: Satu sample = satu production cycle record.
4. Raw Shape: 6000 rows x 19 columns
5. Axis Semantics: Axis 0 = sample/record, Axis 1 = feature/variable
6. Column Roles:
   - Identifier: `record_id`
   - Grouping variable: `machine_id`
   - Temporal/context information: `timestamp`, `operating_hours`, `shift`
   - Numerical features: `operating_hours`, `ambient_temp_c`, `machine_temp_c`, `pressure_bar`, `vibration_mm_s`, `spindle_speed_rpm`, `motor_current_a`, `humidity_pct`, `cycle_time_s`, `tool_wear_pct`
   - Categorical features: `machine_type`, `shift`, `operator_id`, `material_type`
   - Target: `quality_status` (Good / Defect)
   - Additional non-feature column (excluded): `machine_condition` (informational / alternative label)
7. Target: `quality_status` (Binary: Good / Defect)
8. Missing-Value Rules (observed >0% missing):
   - `operator_id`: 1.783% missing → impute with most frequent / `Unknown` for categorical pipeline.
   - `ambient_temp_c`: 3.5% missing → impute with median for numerical pipeline.
   - `pressure_bar`: 4.5% missing → impute with median for numerical pipeline.
   - `vibration_mm_s`: 2.5% missing → impute with median for numerical pipeline.
   - `motor_current_a`: 3.0% missing → impute with median for numerical pipeline.
   - `humidity_pct`: 6.0% missing → impute with median for numerical pipeline.

   Imputation strategy: fit imputers on TRAIN only; validation/test use trained transformers.
9. Encoding Rules: OneHotEncoder(handle_unknown="ignore") applied to categorical features; fit on TRAIN only.
10. Scaling Rules: StandardScaler() applied to numerical features; fit on TRAIN only.
11. Splitting Rules: Group-aware split by `machine_id` to avoid leakage. Target split proportions aimed: ~70% TRAIN, ~15% VALIDATION, ~15% TEST (group-aware proportions used in practice).
12. Leakage-Control Rules: Exclude `record_id`, `machine_id`, and `machine_condition` from feature matrix. Perform split prior to any fit/transform.

Validation rules:
- Preprocessing transformers (imputer, encoder, scaler) MUST be fit only on training data.
- Post-transform features must contain no NaN or infinite values.
- Group overlap between splits must be zero.
