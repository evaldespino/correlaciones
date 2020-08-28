import itertools as itt
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, normalize, scale


np.set_printoptions(precision=3)


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
        self.data = self.data.select_dtypes(include=np.number)
        self.col_names = self.data.columns.to_list()
        self.col_info = {key: index for index, key in enumerate(self.data.columns)}
        self.data = self.data.to_numpy()

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
            return normalize(data, axis=0)
        elif mode == "scale":
            return scale(data)
        else:
            raise ValueError(
                f"Was expecting one of {{'normalize', 'scale'}}, {mode} was received."
            )

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    def transform_x(self, *args, **kwargs):
        raise NotImplementedError

    def _get_extended_result(self, model, X, y, var_indexes: list) -> dict:
        funcs = {
            "r2": lambda model, X, y: model.score(X, y),
            "f_values": lambda model, X, y: f_regression(X, y)[0],
            "p_values": lambda model, X, y: f_regression(X, y)[1],
            "cv_r2": lambda model, X, y: cross_val_score(
                estimator=model, X=X, y=y, cv=2
            ),
            "intercept": lambda *args: model.intercept_,
            "coef": lambda *args: model.coef_,
            "regressors": lambda *args: [self.col_names[idx] for idx in var_indexes],
            # "regressors": lambda *args: var_indexes
        }
        # d = {}
        # for key, func in funcs.items():
        #     d[key] = func(model, X, y)
        return {key: func(model, X, y) for key, func in funcs.items()}
        # r_2 = model.score(X, y)
        # regressors = [self.col_names[idx] for idx in var_idxs]
        # scores = cross_val_score(estimator=model, X=X, y=y, cv=2)
        # f_values, p_values = f_regression(X, y)
        # result = (
        #     r_2,
        #     f_values,
        #     p_values,
        #     scores,
        #     model.intercept_,
        #     model.coef_,
        #     regressors,
        # )
        # return result


class DescriptorCorrelation(CorrelationBase):
    def set_params(self, ignore=None, r_ref: float = 0):
        super()._select_data()
        self.r_ref = r_ref
        choose_r = 2
        indexes = np.arange(self.data.shape[1])
        self.combinations = list(itt.combinations(indexes, r=choose_r))
        # self.combinations = list(itt.combinations(self.col_names, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            x_index, y_index = current_comb
            X, y = self.prepare_data(
                x=self.data[:, x_index],
                y=self.data[:, y_index],
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
    def set_params(self, prop_name: str, desc_num: int = 2, r_ref: float = 0):
        super()._select_data()
        self.desc_num = desc_num
        self.r_ref = r_ref
        # prop_idx = self.data.get_loc(prop_name)
        prop_idx = self.col_info[prop_name]
        self.property = self.data[:, prop_idx]
        choose_r = desc_num if desc_num <= len(self.col_names) else 2  # TODO: Check
        indexes = np.arange(self.data.shape[1])
        indexes = indexes[indexes != prop_idx]
        self.combinations = list(itt.combinations(indexes, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                X=self.data[:, current_comb], preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.property)
            r_2 = model.score(X, y=self.property)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.property, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, X, preprocessing):
        X = super()._preprocess(data=X, mode=preprocessing)
        return X


class PolynomialCorrelation(CorrelationBase):
    def set_params(self, prop_name: str, degree: int, r_ref: float = 0):
        super()._select_data()
        self.degree = degree
        self.r_ref = r_ref
        prop_idx = self.col_info[prop_name]
        self.property = self.data[:, prop_idx]
        choose_r = 1
        indexes = np.arange(self.data.shape[1])
        indexes = indexes[indexes != prop_idx]
        self.combinations = list(itt.combinations(indexes, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data[:, current_comb], preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.property)
            r_2 = model.score(X, y=self.property)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.property, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing):
        X = self.transform_x(x)
        X = PolynomialFeatures(degree=self.degree, include_bias=False).fit_transform(X)
        X = super()._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        has_single_column = data.size == self.data.shape[0]
        return data.reshape((-1, 1)) if has_single_column else data


class PowerCorrelation(CorrelationBase):
    def set_params(self, prop_name: str, n_pow: int = 1, r_ref: int = 0):
        super()._select_data()
        self.n_pow = n_pow
        self.r_ref = r_ref
        prop_idx = self.col_info[prop_name]
        self.property = self.data[:, prop_idx]
        choose_r = 1
        indexes = np.arange(self.data.shape[1])
        indexes = indexes[indexes != prop_idx]
        self.combinations = list(itt.combinations(indexes, r=choose_r))

    def correlation(self, preprocessing=None):
        res = []
        for current_comb in self.combinations:
            X = self.prepare_data(
                x=self.data[:, current_comb], preprocessing=preprocessing
            )
            model = LinearRegression().fit(X, y=self.property)
            r_2 = model.score(X, y=self.property)
            if r_2 >= self.r_ref:
                result = self._get_extended_result(
                    model=model, X=X, y=self.property, var_indexes=current_comb
                )
                res.append(result)
        self.results = pd.DataFrame(res)

    def prepare_data(self, x, preprocessing):
        X = self.transform_x(x)
        X = super()._preprocess(data=X, mode=preprocessing)
        return X

    def transform_x(self, data):
        return data ** self.n_pow
