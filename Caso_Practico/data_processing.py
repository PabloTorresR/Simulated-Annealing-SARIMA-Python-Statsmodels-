import pandas as pd
from typing import Tuple


def data_loader(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .assign(DATE=lambda x: pd.to_datetime(x.DATE))
        .set_index("DATE")
    )


def separate_train_test(
    series: pd.DataFrame, test_cases: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_training = series.iloc[:-test_cases, :]
    data_test = series.iloc[-test_cases:, :]
    return data_training, data_test
