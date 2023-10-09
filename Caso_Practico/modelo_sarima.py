from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import nsdiffs
from statsmodels.stats.diagnostic import acorr_ljungbox

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from grafo import Coeficients


class SARIMAModel:
    def __init__(self, series: pd.Series, coeficients: Coeficients):
        self.series = series
        self.coeficients = coeficients

    def fit_model(self):
        model = SARIMAX(
            self.series,
            order=(self.coeficients.p, self.coeficients.d, self.coeficients.q),
            seasonal_order=(
                self.coeficients.P,
                self.coeficients.D,
                self.coeficients.Q,
                self.coeficients.m,
            ),
        )
        return model.fit()

    def __get_rmse__(self, test_series: pd.Series, forecasted_values) -> float:
        rmse = np.sqrt(mean_squared_error(test_series, forecasted_values))
        return float(rmse)

    def __get_res_corr_penalization__(
        self,
        model_results,
        lags: int,
        penalization: float = 10,
        threshold: float = 0.05,
    ) -> float:
        test_result = acorr_ljungbox(model_results.resid, lags)  # type: ignore
        # El valor [1] del resultado del test nos indica la significancia estadística de la autocorrelación en cada retraso
        p_values = test_result["lb_pvalue"]
        return penalization if any(p_value < threshold for p_value in p_values) else 0

    def get_test_cost(self, test_series: pd.Series):
        """
        El coste es la suma del rmse sobre el conjunto de test y una penalizacion
        para las soluciones donde los residuos esteén correlacionados
        """
        model_results = self.fit_model()
        forecast = model_results.get_forecast(steps=len(test_series))  # type: ignore
        forecasted_values = forecast.predicted_mean

        rmse = self.__get_rmse__(test_series, forecasted_values)
        cost = self.__get_res_corr_penalization__(model_results, 6, 10, 0.05)
        return rmse + cost


class CoeficientsTest:
    def __init__(self, series: pd.Series):
        self.series = series

    def stationary_test(self) -> int:
        """
        La hipotesis nula es que la serie es no estacionaria (crece, decrece)
        Si dfuller_results[0] > nivel de significancia (0.01, 0.05, 0.1...), entonces es no estacionaria
        Si se demuestra esto, d != 0
        Usaremos significancia del 0.05
        """
        dfuller_results = adfuller(self.series)
        return (
            1 if dfuller_results[1] >= 0.05 else 0
        )  # en primera instancia daremos un valor de 1 a "d"

    def stational_test(self, m) -> int:
        D = nsdiffs(self.series, m=m)
        return D

    def run_all_tests(self, m):
        return self.stationary_test(), self.stational_test(m)
