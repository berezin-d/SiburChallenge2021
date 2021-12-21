
import pathlib
import pandas as pd
from catboost import CatBoostRegressor

MODEL_FILE = pathlib.Path(__file__).parent.joinpath("baseline_model.cbm")
AGG_COLS = ["material_code", "company_code", "country", "region", "manager_code"]

def get_features(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Вычисление признаков для `month`."""

    start_period = month - pd.offsets.MonthBegin(6)
    end_period = month - pd.offsets.MonthBegin(1)

    df = df.loc[:, :end_period]

    features = pd.DataFrame([], index=df.index)
    features["month"] = month.month
    features[[f"vol_tm{i}" for i in range(6, 0, -1)]] = df.loc[:, start_period:end_period].copy()

    rolling = df.rolling(12, axis=1, min_periods=1)
    features = features.join(rolling.mean().iloc[:, -1].rename("last_year_avg"))
    features = features.join(rolling.min().iloc[:, -1].rename("last_year_min"))
    features = features.join(rolling.max().iloc[:, -1].rename("last_year_max"))
    features["month"] = month.month
    return features.reset_index()


def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий все сделки с начала тренировочного периода до `month`; типы
          колонок совпадают с типами в ноутбуке `[SC2021] Baseline`,
        month:
          месяц, для которого вычисляются предсказания.

    Результат:
        Датафрейм предсказаний для каждой группы, содержащий колонки:
            - `material_code`, `company_code`, `country`, `region`, `manager_code`,
            - `prediction`.
        Предсказанные значения находятся в колонке `prediction`.
    """

    group_ts = df.groupby(AGG_COLS + ["month"])["volume"].sum().unstack(fill_value=0)
    features = get_features(group_ts, month)

    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)
    predictions = model.predict(features[model.feature_names_])

    preds_df = features[AGG_COLS].copy()
    preds_df["prediction"] = predictions
    return preds_df
    
