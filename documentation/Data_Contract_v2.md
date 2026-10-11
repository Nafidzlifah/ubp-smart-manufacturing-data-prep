# Data Contract v2 — UBP Smart Manufacturing (Model-Ready Data)

1. Final Sample Definition: Preprocessed production cycle record ready for ML/DL.
2. Final Train Shape: 4163 rows x 52 columns
3. Final Validation Shape: 585 rows x 52 columns
4. Final Test Shape: 1252 rows x 52 columns
5. Final Feature Count: 51 features (excluding target)
6. Final Feature Names: operating_hours, ambient_temp_c, machine_temp_c, pressure_bar, vibration_mm_s, spindle_speed_rpm, motor_current_a, humidity_pct, cycle_time_s, tool_wear_pct, machine_type_CNC-A, machine_type_CNC-B, machine_type_CNC-C, shift_Malam, shift_Pagi, shift_Siang, operator_id_OP-01, operator_id_OP-02, operator_id_OP-03, operator_id_OP-04, operator_id_OP-05, operator_id_OP-06, operator_id_OP-07, operator_id_OP-08, operator_id_OP-09, operator_id_OP-10, operator_id_OP-11, operator_id_OP-12, operator_id_OP-13, operator_id_OP-14, operator_id_OP-15, operator_id_OP-16, operator_id_OP-17, operator_id_OP-18, operator_id_OP-19, operator_id_OP-20, operator_id_OP-21, operator_id_OP-22, operator_id_OP-23, operator_id_OP-24, operator_id_OP-25, operator_id_OP-26, operator_id_OP-27, operator_id_OP-28, operator_id_OP-29, operator_id_OP-30, operator_id_Unknown, material_type_AL-6061, material_type_SPECIAL-X, material_type_SS-304, material_type_STEEL-C45
7. Target: `quality_status`
8. Axis Semantics: Axis 0 = preprocessed sample, Axis 1 = encoded & scaled numerical features + one-hot encoded categorical features
9. Encoding Result: OneHotEncoder applied; categorical features one-hot encoded (train-fit only). Output columns for `machine_type`, `shift`, `operator_id`, `material_type` present as `machine_type_*`, `shift_*`, `operator_id_*`, `material_type_*`.
10. Scaling Result: StandardScaler applied to numerical features; fitted on train only.
11. Missing-Value Result: 0 NaN values in final feature matrices (imputation applied during preprocessing).
12. Split Result:

| split | n_samples | pct | n_groups | groups (partial list) |
|---|---:|---:|---:|---|
| train | 4163 | 69.38 | 14 | MC-01,MC-02,MC-03,MC-04,MC-06,MC-08,MC-09,MC-10,MC-14,MC-15,MC-16,MC-17,MC-18,MC-19 |
| validation | 585 | 9.75 | 2 | MC-13,MC-20 |
| test | 1252 | 20.87 | 4 | MC-05,MC-07,MC-11,MC-12 |

13. Group Leakage Result: No group overlap between splits (TRAIN ∩ VALIDATION = ∅, TRAIN ∩ TEST = ∅, VALIDATION ∩ TEST = ∅) as per `evidence/split_evidence.csv`.
14. Sanity Check Result: SANITY CHECK: PASSED
15. Reproducibility: Pipeline used fixed random seed 2022 for group-aware splitting; preprocessing implemented with scikit-learn `Pipeline` and `ColumnTransformer` (imputer on numeric: median; imputer on categorical: constant 'Unknown'; OneHotEncoder(handle_unknown='ignore'); StandardScaler()).

Notes:
- Final feature count (51) excludes the target column `quality_status`.
- All numeric features were scaled and categorical features one-hot encoded; final CSVs are in `output/`.
