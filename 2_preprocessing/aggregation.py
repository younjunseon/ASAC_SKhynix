"""
Die → Unit 집계 모듈
- utils/aggregate.py의 함수를 활용하여 전처리 완료된 데이터를 집계
- 집계 결과를 CSV로 저장

EDA 결과 기반:
- 모든 unit은 정확히 4개 die로 구성 (position 1~4)
- 단일 feature 상관 max |r|=0.037 → 집계 통계량의 다양성이 핵심
"""
import pandas as pd
import numpy as np
import os
import sys

# utils 경로 등록
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.config import KEY_COL, TARGET_COL, OUTPUT_DIR
from utils.aggregate import aggregate_to_unit, pivot_by_position, merge_with_target


def run_aggregation(xs_train, xs_val, xs_test, feat_cols,
                    agg_funcs=None, use_position_pivot=False,
                    save_csv=True, save_format="csv", output_dir=None):
    """
    전처리 완료된 die-level 데이터를 unit-level로 집계

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
        클리닝 + 이상치 처리 완료된 die-level 데이터
    feat_cols : list
    agg_funcs : list of str
        기본: ["mean", "std", "min", "max"]
    use_position_pivot : bool
        True면 position별 피벗도 추가
    save_csv : bool
        True면 집계 결과를 디스크에 저장 (플래그 이름은 하위호환).
        실제 포맷은 save_format으로 선택.
    save_format : {"csv", "parquet"}
        저장 포맷. 기본 "csv" (하위호환).
        "parquet"이면 pyarrow 엔진으로 저장. CSV 대비 파일 크기 ~1/5,
        읽기/쓰기 5~10배 빠름.
    output_dir : str
        저장 경로. None이면 OUTPUT_DIR 사용

    Returns
    -------
    unit_train, unit_val, unit_test : DataFrame (unit-level)
    unit_feat_cols : list (집계 후 feature 컬럼)
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max", "range", "median"]
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("=" * 60)
    print("Die → Unit 집계 시작")
    print(f"  agg_funcs: {agg_funcs}")
    print(f"  position_pivot: {use_position_pivot}")
    print("=" * 60)

    # 입력 dtype 감지 (float32 다운캐스트 파이프라인 여부 판별)
    in_dtype = None
    try:
        in_dtype = xs_train[feat_cols].dtypes.iloc[0]
    except (AttributeError, IndexError):
        pass

    # 각 split별 집계
    parts = {}
    for name, xs_split in [("train", xs_train), ("validation", xs_val), ("test", xs_test)]:
        agg_df = aggregate_to_unit(xs_split, feat_cols, agg_funcs)

        if use_position_pivot:
            pos_df = pivot_by_position(xs_split, feat_cols)
            agg_df = agg_df.join(pos_df, how="left")

        # groupby.agg는 보통 dtype 보존하지만 일부 버전에서 float64로 승격될 수 있어
        # 입력이 float32면 결과도 float32로 강제 (메모리 일관성)
        if in_dtype == np.float32 and not agg_df.empty:
            float_cols = agg_df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                agg_df[float_cols] = agg_df[float_cols].astype('float32')

        parts[name] = agg_df

    unit_train = parts["train"]
    unit_val = parts["validation"]
    unit_test = parts["test"]
    unit_feat_cols = unit_train.columns.tolist()

    print(f"\n집계 결과:")
    print(f"  train: {unit_train.shape}")
    print(f"  val:   {unit_val.shape}")
    print(f"  test:  {unit_test.shape}")
    print(f"  feature 수: {len(unit_feat_cols)}")

    # 디스크 저장
    if save_csv:
        if save_format not in ("csv", "parquet"):
            raise ValueError(
                f"save_format must be 'csv' or 'parquet', got {save_format!r}"
            )

        os.makedirs(output_dir, exist_ok=True)
        ext = "csv" if save_format == "csv" else "parquet"
        files = {
            f"unit_train.{ext}": unit_train,
            f"unit_val.{ext}":   unit_val,
            f"unit_test.{ext}":  unit_test,
        }
        for fname, df in files.items():
            path = os.path.join(output_dir, fname)
            if save_format == "csv":
                df.to_csv(path)
            else:
                # parquet은 index 보존이 명시적이어야 함
                df.to_parquet(path, engine="pyarrow", index=True)

        print(f"\n저장 완료 → {output_dir}/unit_*.{ext}")

        # Colab: 브라우저로 다운로드
        from utils.config import ENV
        if ENV == "colab":
            from google.colab import files as colab_files
            for fname in files:
                colab_files.download(os.path.join(output_dir, fname))
            print("Colab: 브라우저 다운로드 실행")

    print("=" * 60)
    return unit_train, unit_val, unit_test, unit_feat_cols
