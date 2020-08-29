import itertools as itt
import sys
from typing import Iterable, List, Union

import pandas as pd
import sklearn.preprocessing as skprep
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from util._validators import validate_sequence


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

    def restrict(self, conditions: list):
        conditions = self._format_restrictions(conditions)
        self.data = self.data[eval(conditions)]

    @staticmethod
    def _format_restrictions(conditions: list) -> str:
        fmt_str = ""
        for column, condition, operator in conditions:
            fmt_str += f"(self.data[{column}] {condition})"
            if operator == "-1":
                break
            fmt_str = f"{fmt_str} {operator} "
        return fmt_str

    def _select_data(self):
        self.data = self.data.select_dtypes(include="number")

    def set_params(self, *args, **kwargs):
        funcname = sys._getframe().f_code.co_name
        raise NotImplementedError(f"{funcname} must be implemented by subclass.")

    def correlation(self, *args, **kwargs):
        funcname = sys._getframe().f_code.co_name
        raise NotImplementedError(f"{funcname} must be implemented by subclass.")

    def print_results(
        self, limit: int = None, sortby: str = "r2", ascending: bool = False
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
            title = self._make_title(result.regressors)
            print(
                f"{title} R2: {result.r2:.3f} Cv_R2: {result.cv_r2[0]:.3f}",
                f"Ordenada: {result.intercept:.3f} Coef: {result.coef}",
                f"F: {result.f_values}\n",
            )

    def _make_title(self, regressors: list) -> str:
        title = ""
        for regressor in regressors:
            title = f"{title} {regressor}"
            if hasattr(self, "n_pow") and self.n_pow != 1:
                title += f"**{self.n_pow}"
        return title

    def save_to_csv(self, filepath: str, sep: str = ","):
        pass

    @staticmethod
    def _preprocess(data, mode: str = None):
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

    def _get_extended_result(self, model, X, y, var_indexes: list) -> dict:
        f_values, p_values = f_regression(X, y)
        result = {
            "r2": model.score(X, y),
            "f_values": f_values,
            "p_values": p_values,
            "cv_r2": cross_val_score(estimator=model, X=X, y=y, cv=2),
            "intercept": model.intercept_,
            "coef": model.coef_,
            "regressors": var_indexes,
        }
        return result

    @staticmethod
    def _drop_from_index(index, ignore: list):
        if ignore is not None:
            index = index.drop(validate_sequence(ignore))
        return index


class DescriptorCorrelation(CorrelationBase):
    def set_params(self, ignore: Union[str, Iterable] = None, r_ref: float = 0):
        super()._select_data()
        self.r_ref = r_ref
        choose_r = 2
        pool = super()._drop_from_index(index=self.data.columns, ignore=ignore)
        self.combinations = list(itt.combinations(pool, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            x_index, y_index = current_comb
            X, y = self.prepare_data(
                x=self.data.loc[:, x_index].to_numpy(),
                y=self.data.loc[:, y_index].to_numpy(),
                preprocessing=preprocessing,
            )
            model = LinearRegression().fit(X, y)
            r_2 = model.score(X, y)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=y, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, y, preprocessing):
        X = self.transform_x(x)
        X = super()._preprocess(data=X, mode=preprocessing)
        return X, y

    def transform_x(self, data):
        return data.reshape((-1, 1))


class PropertiesCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        desc_num: int = 2,
        ignore: Union[str, Iterable] = None,
        r_ref: float = 0,
    ):
        super()._select_data()
        self.desc_num = desc_num
        self.r_ref = r_ref
        self.target = self.data[target].to_numpy()
        choose_r = desc_num if desc_num <= len(self.data.columns) else 2  # TODO: Check
        pool = super()._drop_from_index(index=self.data.columns, ignore=target)
        pool = super()._drop_from_index(index=pool, ignore=ignore)
        self.combinations = list(itt.combinations(pool, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                X=self.data.loc[:, current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.target, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, X, preprocessing):
        X = super()._preprocess(data=X, mode=preprocessing)
        return X


class PolynomialCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        degree: int,
        ignore: Union[str, Iterable] = None,
        r_ref: float = 0,
    ):
        super()._select_data()
        self.degree = degree
        self.r_ref = r_ref
        self.target = self.data[target].to_numpy()
        choose_r = 1
        pool = super()._drop_from_index(index=self.data.columns, ignore=target)
        pool = super()._drop_from_index(index=pool, ignore=ignore)
        self.combinations = list(itt.combinations(pool, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data.loc[:, current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.target, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing):
        X = self.transform_x(x)
        X = skprep.PolynomialFeatures(
            degree=self.degree, include_bias=False
        ).fit_transform(X)
        X = super()._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        has_single_column = data.size == self.data.shape[0]
        return data.reshape((-1, 1)) if has_single_column else data


class PowerCorrelation(CorrelationBase):
    def set_params(
        self,
        target: str,
        n_pow: int = 1,
        ignore: Union[str, Iterable] = None,
        r_ref: int = 0,
    ):
        super()._select_data()
        self.n_pow = n_pow
        self.r_ref = r_ref
        self.target = self.data[target].to_numpy()
        choose_r = 1
        pool = super()._drop_from_index(index=self.data.columns, ignore=target)
        pool = super()._drop_from_index(index=pool, ignore=ignore)
        self.combinations = list(itt.combinations(pool, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data.loc[:, current_comb].to_numpy(), preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.target)
            r_2 = model.score(X, y=self.target)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.target, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing):
        X = self.transform_x(x)
        X = super()._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        return data ** self.n_pow
