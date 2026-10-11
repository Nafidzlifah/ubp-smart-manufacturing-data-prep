"""
Simple data profiling script for production_tabular_raw.csv
Generates evidence/data_profile.csv and evidence/outlier_summary.csv
"""
import os
import pandas as pd
import numpy as np


def main():
    root = os.path.dirname(__file__)
    raw_path = os.path.join(root, 'production_tabular_raw.csv')
    out_dir = os.path.join(root, 'evidence')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    profile = []
    profile.append(('n_rows', df.shape[0]))
    profile.append(('n_columns', df.shape[1]))
    profile.append(('columns', ','.join(df.columns.tolist())))
    types = df.dtypes.apply(lambda x: x.name).to_dict()
    profile.append(('dtypes', types))
    missing_count = df.isna().sum().to_dict()
    profile.append(('missing_count', missing_count))
    missing_pct = (df.isna().mean() * 100).round(3).to_dict()
    profile.append(('missing_pct', missing_pct))
    dup_count = df.duplicated().sum()
    profile.append(('duplicate_count', int(dup_count)))
    unique_counts = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    profile.append(('unique_counts', unique_counts))

    # basic numeric stats
    numeric = df.select_dtypes(include=[np.number])
    num_stats = numeric.describe().T
    num_stats.to_csv(os.path.join(out_dir, 'numerical_stats.csv'))

    # categorical distributions for low-cardinality
    cat_cols = [
        column for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ]
    cat_dist = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(20).to_dict()
        cat_dist[c] = vc

    # target distribution
    if 'quality_status' in df.columns:
        target_dist = df['quality_status'].value_counts(dropna=False).to_dict()
    else:
        target_dist = {}

    # machine distribution
    if 'machine_id' in df.columns:
        machine_dist = df['machine_id'].value_counts().to_dict()
    else:
        machine_dist = {}

    profile_rows = [
        ['n_rows', df.shape[0]],
        ['n_columns', df.shape[1]],
        ['duplicate_count', int(dup_count)],
    ]

    profile_df = pd.DataFrame(profile_rows, columns=['metric', 'value'])
    profile_df.to_csv(os.path.join(out_dir, 'data_profile.csv'), index=False)

    # save detailed items
    pd.Series(types).to_frame('dtype').to_csv(os.path.join(out_dir, 'column_types.csv'))
    pd.Series(missing_count).to_frame('missing_count').to_csv(os.path.join(out_dir, 'missing_count.csv'))
    pd.Series(missing_pct).to_frame('missing_pct').to_csv(os.path.join(out_dir, 'missing_pct.csv'))
    pd.Series(unique_counts).to_frame('unique_counts').to_csv(os.path.join(out_dir, 'unique_counts.csv'))

    # save categorical distributions
    import json
    with open(os.path.join(out_dir, 'categorical_distribution.json'), 'w', encoding='utf8') as f:
        json.dump(cat_dist, f, indent=2)
    with open(os.path.join(out_dir, 'target_distribution.json'), 'w', encoding='utf8') as f:
        json.dump(target_dist, f, indent=2)
    with open(os.path.join(out_dir, 'machine_distribution.json'), 'w', encoding='utf8') as f:
        json.dump(machine_dist, f, indent=2)

    # outlier summary (IQR rule)
    outlier_rows = []
    for col in numeric.columns:
        ser = numeric[col].dropna()
        if ser.empty:
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        n_low = int((ser < low).sum())
        n_high = int((ser > high).sum())
        outlier_rows.append([col, int(ser.count()), n_low, n_high, float(low), float(high)])

    outlier_df = pd.DataFrame(outlier_rows, columns=['column', 'count', 'n_low', 'n_high', 'low_threshold', 'high_threshold'])
    outlier_df.to_csv(os.path.join(out_dir, 'outlier_summary.csv'), index=False)

    print('Profiling complete. Outputs in evidence/')


if __name__ == '__main__':
    main()
