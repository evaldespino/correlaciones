import sys
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import sklearn.preprocessing as skprep
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from _validators import validate_sequence
from utils import sequential_key_dict, list_combinations


pd.set_option("display.precision", 3)


class CorrelationBase:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError
        self.data = data

    @classmethod
    def from_csv(cls, filepath: str, sep: str = ","):
        data = pd.read_csv(filepath, sep)
        return cls(data)

    def restrict(self, conditions: List[str]):
        conditions = self._format_restrictions(conditions)
        self.data = self.data[eval(conditions)]

    @staticmethod
    def _format_restrictions(conditions: List[str]) -> str:
        fmt_str = ""
        for column, condition, operator in conditions:
            fmt_str += f"(self.data[{column}] {condition})"
            if operator == "-1":
                break
            fmt_str = f"{fmt_str} {operator} "
        return fmt_str

    def set_params(
        self,
        target: Optional[str] = None,
        ignore: Optional[Union[str, Iterable]] = None,
        r_ref: float = 0,
    ):
        self.data = self.data.select_dtypes(include="number")
        self.r_ref = r_ref
        if target is not None:
            self.target_name = target
            self.target = self.data[target].to_numpy()
        self.pool = self.__drop_from_index(index=self.data.columns, ignore=target)
        self.pool = self.__drop_from_index(index=self.pool, ignore=ignore)

    def correlation(self, *args, **kwargs):
        funcname = sys._getframe().f_code.co_name
        raise NotImplementedError(f"{funcname} must be implemented by subclass.")

    def print_results(
        self, limit: Optional[int] = None, sortby: str = "r2", ascending: bool = False
    ):
        if not len(self.results):
            print(f"No se encontraron correlaciones con R2 mayor a {self.r_ref}")
            return None
        if sortby not in self.results.columns:
            sortby = "r2"
        results = self.results.sort_values(
            by=sortby, kind="mergesort", ascending=ascending, ignore_index=True
        )
        limit = limit or float("inf")
        for result in results.itertuples():
            if result.Index >= limit:
                break
            title = self._make_title(result.regressors, result.regressand)
            print(
                f"{title} R2: {result.r2:.3f} Cv_R2: {result.cv_r2[0]:.3f}",
                f"Ordenada: {result.intercept:.3f} Coef: {result.coef}",
                f"F: {result.f_values}\n",
            )

    def _make_title(self, *args, **kwargs):
        raise NotImplementedError

    def save_to_csv(self, filepath: str, sep: str = ","):
        pass

    @staticmethod
    def _preprocess(data, mode: Optional[str] = None):
        if mode is None:
            return data
        elif mode == "normalize":
            return skprep.normalize(data, axis=0)
        elif mode == "scale":
            return skprep.scale(data)
        else:
            raise ValueError(
                f"Was expecting one of {{'normalize', 'scale'}}, {mode} was received."
            )

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    def transform_x(self, *args, **kwargs):
        raise NotImplementedError

    def _get_extended_result(
        self, model, X, y, regressor_names: List[str], regressand_name: str
    ) -> Dict[str, Any]:
        f_values, p_values = f_regression(X, y)
        result = {
            "r2": model.score(X, y),
            "f_values": f_values,
            "p_values": p_values,
            "cv_r2": cross_val_score(estimator=model, X=X, y=y, cv=2),
            "intercept": model.intercept_,
            "coef": model.coef_,
            "regressors": regressor_names,
            "regressand": regressand_name,
        }
        return result

    @staticmethod
    def __drop_from_index(index, ignore: List[str]):
        if ignore is not None:
            index = index.drop(validate_sequence(ignore))
        return index


class DescriptorCorrelation(CorrelationBase):
    def set_params(
        self, ignore: Optional[Union[str, Iterable]] = None, r_ref: float = 0
    ):
        super().set_params(target=None, ignore=ignore, r_ref=r_ref)
        choose_r = 2
        self.combinations = list_combinations(self.pool, r=choose_r)

    def correlation(self, preprocessing: Optional[str] = None):
        res = []
        for current_comb in self.combinations:
            x_name, y_name = current_comb
            X, y = self.prepare_data(
                x=self.data[x_name].to_numpy(),
                y=self.data[y_name].to_numpy(),
                preprocessing=preprocessing,
            )
            model = LinearRegression().fit(X, y)
            r_2 = model.score(X, y)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model,
                    X=X,
                    y=y,
                    regressor_names=x_name,
                    regressand_name=y_name,
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, y, preprocessing: Optional[str]):
        X = self.transform_x(x)
        X = self._preprocess(data=X, mode=preprocessing)
        return X, y

    def transform_x(self, data):
        return data.reshape((-1, 1))

    @staticmethod
    def _make_title(regressor: str, regressand: str) -> str:
        return f"{regressand} ~ {regressor}"


class PropertiesCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        desc_num: int = 2,
        ignore: Optional[Union[str, Iterable]] = None,
        r_ref: float = 0,
    ):
        super().set_params(target=target, ignore=ignore, r_ref=r_ref)
        self.desc_num = desc_num
        choose_r = desc_num if desc_num <= len(self.data.columns) else 2  # TODO: Check
        self.combinations = list_combinations(self.pool, r=choose_r)

    def correlation(self, preprocessing: Optional[str] = None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                X=self.data[current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model,
                    X=X,
                    y=self.target,
                    regressor_names=current_comb,
                    regressand_name=self.target_name,
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, X, preprocessing: Optional[str]):
        X = self._preprocess(data=X, mode=preprocessing)
        return X

    @staticmethod
    def _make_title(regressors: Iterable, regressand: str) -> str:
        regressors_str = " + ".join(regressors)
        return f"{regressand} ~ {regressors_str}"


class PolynomialCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        degree: int,
        ignore: Optional[Union[str, Iterable]] = None,
        r_ref: float = 0,
    ):
        super().set_params(target=target, ignore=ignore, r_ref=r_ref)
        self.degree = degree
        choose_r = 1
        self.combinations = list_combinations(self.pool, r=choose_r)

    def correlation(self, preprocessing: Optional[str] = None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data[current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model,
                    X=X,
                    y=self.target,
                    regressor_names=current_comb,
                    regressand_name=self.target_name,
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing: Optional[str]):
        X = self.transform_x(x)
        X = skprep.PolynomialFeatures(
            degree=self.degree, include_bias=False
        ).fit_transform(X)
        X = self._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        has_single_column = data.size == self.data.shape[0]
        return data.reshape((-1, 1)) if has_single_column else data

    def _make_title(self, regressor: Iterable, regressand: str) -> str:
        regressor_str = regressor[0]
        return f"{regressand} ~ O({regressor_str}^{self.degree})"


class PowerCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        n_pow: int = 1,
        ignore: Optional[Union[str, Iterable]] = None,
        r_ref: float = 0,
    ):
        super().set_params(target=target, ignore=ignore, r_ref=r_ref)
        self.n_pow = n_pow
        choose_r = 1
        self.combinations = list_combinations(self.pool, r=choose_r)

    def correlation(self, preprocessing: Optional[str] = None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data[current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model,
                    X=X,
                    y=self.target,
                    regressor_names=current_comb,
                    regressand_name=self.target_name,
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing: Optional[str]):
        X = self.transform_x(x)
        X = self._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        return data ** self.n_pow

    def _make_title(self, regressor: Iterable, regressand: str) -> str:
        regressor_str = regressor[0]
        return f"{regressand} ~ {regressor_str}^{self.n_pow}"
