"""
Preprocessing pipeline script.
Performs group-aware split and fits preprocessing pipeline on training data.
Outputs model-ready CSVs under output/ and evidence/split_evidence.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


ROOT = os.path.dirname(__file__)
RAW = os.path.join(ROOT, 'production_tabular_raw.csv')
OUT_DIR = os.path.join(ROOT, 'output')
EVID = os.path.join(ROOT, 'evidence')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EVID, exist_ok=True)


def load_data():
    return pd.read_csv(RAW)


def define_columns(df):
    id_cols = ['record_id']
    group_col = 'machine_id'
    temporal = ['timestamp']
    target = 'quality_status'
    # candidate features
    numeric = ['operating_hours','ambient_temp_c','machine_temp_c','pressure_bar','vibration_mm_s','spindle_speed_rpm','motor_current_a','humidity_pct','cycle_time_s','tool_wear_pct']
    categorical = ['machine_type','shift','operator_id','material_type']
    return id_cols, group_col, temporal, target, numeric, categorical


def group_split(df, group_col='machine_id', seed=2022, train_size=0.7, val_size=0.15):
    # first split train vs temp (val+test)
    groups = df[group_col]
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # split temp into val and test by proportion
    # compute relative sizes
    temp_size = 1.0 - train_size
    val_prop = val_size / temp_size
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed)
    val_idx_rel, test_idx_rel = next(gss2.split(df_temp, groups=df_temp[group_col]))
    df_val = df_temp.iloc[val_idx_rel].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx_rel].reset_index(drop=True)

    return df_train, df_val, df_test


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])
    return preprocessor


def fit_transform_save(df_train, df_val, df_test, id_cols, group_col, target, numeric, categorical):
    X_train = df_train[numeric + categorical]
    y_train = df_train[target]
    X_val = df_val[numeric + categorical]
    y_val = df_val[target]
    X_test = df_test[numeric + categorical]
    y_test = df_test[target]

    pre = build_preprocessor(numeric, categorical)
    pre.fit(X_train)

    X_train_t = pre.transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # assemble back to DataFrame with feature names
    num_names = numeric
    # get categorical output names
    ohe = pre.named_transformers_['cat'].named_steps['onehot']
    cat_names = ohe.get_feature_names_out(categorical).tolist()
    feature_names = num_names + cat_names

    df_train_out = pd.DataFrame(X_train_t, columns=feature_names)
    df_train_out[target] = y_train.values
    df_val_out = pd.DataFrame(X_val_t, columns=feature_names)
    df_val_out[target] = y_val.values
    df_test_out = pd.DataFrame(X_test_t, columns=feature_names)
    df_test_out[target] = y_test.values

    # save
    df_train_out.to_csv(os.path.join(OUT_DIR, 'train_model_ready.csv'), index=False)
    df_val_out.to_csv(os.path.join(OUT_DIR, 'validation_model_ready.csv'), index=False)
    df_test_out.to_csv(os.path.join(OUT_DIR, 'test_model_ready.csv'), index=False)

    return pre


def save_split_evidence(df_train, df_val, df_test, group_col='machine_id', target='quality_status'):
    rows = []
    total = sum([len(df_train), len(df_val), len(df_test)])
    for name, df in [('train', df_train), ('validation', df_val), ('test', df_test)]:
        rows.append({
            'split': name,
            'n_samples': len(df),
            'pct': round(100 * len(df) / total, 2),
            'n_groups': int(df[group_col].nunique()),
            'groups': ','.join(sorted(df[group_col].unique()))[:1000],
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EVID, 'split_evidence.csv'), index=False)


def sanity_check_files():
    msgs = []
    # check model-ready files exist
    paths = [os.path.join(OUT_DIR, f) for f in ['train_model_ready.csv','validation_model_ready.csv','test_model_ready.csv']]
    for p in paths:
        if not os.path.exists(p):
            msgs.append(f'MISSING {p}')
    if msgs:
        with open(os.path.join(EVID, 'sanity_check.txt'), 'w') as f:
            f.write('SANITY CHECK: FAILED\n')
            for m in msgs:
                f.write(m + '\n')
        print('Sanity check failed, see evidence/sanity_check.txt')
    else:
        with open(os.path.join(EVID, 'sanity_check.txt'), 'w') as f:
            f.write('SANITY CHECK: PASSED\n')
        print('Sanity check passed.')


def main():
    df = load_data()
    id_cols, group_col, temporal, target, numeric, categorical = define_columns(df)
    df_train, df_val, df_test = group_split(df, group_col=group_col, seed=2022)

    # save split evidence
    save_split_evidence(df_train, df_val, df_test, group_col=group_col, target=target)

    pre = fit_transform_save(df_train, df_val, df_test, id_cols, group_col, target, numeric, categorical)
    sanity_check_files()


if __name__ == '__main__':
    main()
