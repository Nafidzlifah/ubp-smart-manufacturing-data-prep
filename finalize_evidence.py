"""
Finalize evidence script.

TASK 1: create evidence/model_ready_data_record.csv by sampling rows from
        output train/validation/test model-ready CSVs and adding a source_split column.

TASK 2: run a sanity check across the three model-ready files and write the
        result to evidence/sanity_check.txt (PASSED or FAILED with reasons).
"""
import os
import pandas as pd
import numpy as np


def task1_model_ready_record(root):
    out_dir = os.path.join(root, 'evidence')
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        'train': os.path.join(root, 'output', 'train_model_ready.csv'),
        'validation': os.path.join(root, 'output', 'validation_model_ready.csv'),
        'test': os.path.join(root, 'output', 'test_model_ready.csv'),
    }

    parts = []
    # read required head rows for each split
    head_counts = {'train': 3, 'validation': 2, 'test': 2}
    for split, p in paths.items():
        if os.path.exists(p):
            try:
                n = head_counts[split]
                df = pd.read_csv(p, nrows=n)
                if not df.empty:
                    df = df.copy()
                    df['source_split'] = split
                    parts.append(df)
            except Exception as e:
                # if reading fails, skip sample creation but continue checks
                print(f'Warning: could not read {p} for sample extraction: {e}')
        else:
            print(f'Info: file not found for sample extraction: {p}')

    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = pd.DataFrame()

    out_path = os.path.join(out_dir, 'model_ready_data_record.csv')
    out.to_csv(out_path, index=False)
    print(f'Wrote sample record to {out_path} (rows={len(out)})')


def task2_sanity_check(root):
    out_dir = os.path.join(root, 'evidence')
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(root, 'output', 'train_model_ready.csv'),
        os.path.join(root, 'output', 'validation_model_ready.csv'),
        os.path.join(root, 'output', 'test_model_ready.csv'),
    ]

    reasons = []
    dfs = {}
    col_counts = {}

    for p in paths:
        if not os.path.exists(p):
            reasons.append(f'Missing file: {p}')
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            reasons.append(f'Failed to read {p}: {e}')
            continue

        if df.empty:
            reasons.append(f'File empty: {p}')

        dfs[p] = df
        col_counts[p] = df.shape[1]

        # target existence
        if 'quality_status' not in df.columns:
            reasons.append(f"Missing target 'quality_status' in {p}")

        # feature NaN check (exclude target)
        feature_df = df.drop(columns=['quality_status'], errors='ignore')
        n_missing = int(feature_df.isna().sum().sum())
        if n_missing > 0:
            reasons.append(f'NaN values in features in {p}: {n_missing} total')

        # infinite check on numeric features
        num = feature_df.select_dtypes(include=[np.number])
        if not num.empty:
            arr = num.to_numpy()
            # np.isfinite works on numeric arrays
            finite_mask = np.isfinite(arr)
            if not finite_mask.all():
                n_inf = int((~finite_mask).sum())
                reasons.append(f'Infinite or non-finite numeric values in {p}: {n_inf} entries')

    # column count consistency
    if len(col_counts) > 0:
        counts = set(col_counts.values())
        if len(counts) > 1:
            details = ', '.join([f"{os.path.basename(k)}={v}" for k, v in col_counts.items()])
            reasons.append(f'Inconsistent number of columns across files: {details}')

    # write sanity result
    sanity_path = os.path.join(out_dir, 'sanity_check.txt')
    if not reasons:
        with open(sanity_path, 'w', encoding='utf8') as f:
            f.write('SANITY CHECK: PASSED')
        print('SANITY CHECK: PASSED')
    else:
        with open(sanity_path, 'w', encoding='utf8') as f:
            f.write('SANITY CHECK: FAILED\n')
            for r in reasons:
                f.write(r + '\n')
        print('SANITY CHECK: FAILED')
        for r in reasons:
            print('-', r)


def main():
    root = os.path.dirname(__file__)
    task1_model_ready_record(root)
    task2_sanity_check(root)


if __name__ == '__main__':
    main()
