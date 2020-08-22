# -*- coding: utf-8 -*-
import argparse as ar
import itertools as it
from time import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, normalize, scale


class Corr:
    def __init__(self, path):
        self.path = path
        self.num_des = 0
        self.orden_polinomio = 0
        self.n = 0
        self.res_lim = float("inf")

    def read_csv(self, sep):
        print("Leyendo los datos del CSV")
        file = pd.read_csv(self.path, sep)
        print("Se encontraron", file.shape[1], "Descriptores y", file.shape[0], "datos")
        print("¿Restringir base de datos? \n 1: Sí \n 2: No ")
        res = int(input())
        if res == 1:
            restrictions = self._ask_restrictions()
            file = file[eval(restrictions)]
        file = file.select_dtypes(include=np.number)
        self.noms = file.columns.to_list()
        self.datos = file.to_numpy()
        print("Se encontraron", len(self.noms),
              "Descriptores numéricos y", self.datos.shape[0], "datos")

    def set_params(self):
        print(f"# Columna a Evaluar (0 - {len(self.noms) - 1})")
        for index, col_name in enumerate(self.noms):
            print(f"{index}: {col_name}")
        self.num = int(input())
        print(self.noms[self.num], "será la propiedad a evaluar")
        print("Escoge la rutina:\n 1: Correlación entre descriptores \n 2: Correlación de propiedades \n 3: Correlaciones polinomiales \n 4: Correlaciones con descriptores a la potencia n")
        ent = int(input())
        if ent == 1:
            choose_r = 2
        elif ent == 2:
            choose_r = self.num_des = int(input("Número de descriptores: "))
        elif ent in (3, 4):
            self.num_des = -1
            choose_r = 1
            if ent == 3:
                self.orden_polinomio = int(input("Orden máximo del polinomio: "))
            elif ent == 4:
                self.n = float(input("Potencia n (des**n): "))
        else:
            raise ValueError(f"{ent} no es una opción válida")
        indexes = np.arange(self.datos.shape[1])
        indexes = indexes[indexes != self.num]
        self.comb = np.array(list(it.combinations(indexes, r=choose_r)))
        self.r_ref = float(input("R2 mínima: "))

    def corr(self):
        ent_pre = int(input("Preprocesamiento:\n 1: Ninguno \n 2: Normalizar \n 3: Escalar\n"))
        ent_lim = int(input("Limitar Resultados:\n 1: Sí \n 2: No\n"))
        if ent_lim == 1:
            self.res_lim = int(input("Número máximo de resultados a mostrar: "))
        np.set_printoptions(precision=3)
        ti = time()
        prop = self.datos[:, self.num]
        res = []
        for current_comb in self.comb:
            if self.num_des != 0:
                des_ev = self.datos[:, current_comb]
            else:
                des_ev = self.datos[:, current_comb[1]]
                prop = self.datos[:, current_comb[0]]
            if des_ev.size == self.datos.shape[0]:
                des_ev = des_ev.reshape((-1, 1))
            if self.orden_polinomio != 0:
                des_ev = PolynomialFeatures(
                    degree=self.orden_polinomio, include_bias=False).fit_transform(des_ev)
            if self.n != 0:
                des_ev = des_ev**(self.n)
            if ent_pre == 1:
                des_ev = des_ev
            elif ent_pre == 2:
                des_ev = normalize(des_ev, axis=0)
            elif ent_pre == 3:
                des_ev = scale(des_ev)
            title = ""
            for col_index in current_comb:
                title = f"{title} {self.noms[col_index]}"
                if self.n != 0:
                    title += f"**{self.n}"
            model = LinearRegression().fit(X=des_ev, y=prop)
            r_2 = model.score(X=des_ev, y=prop)
            if r_2 >= self.r_ref:
                scores = cross_val_score(estimator=model, X=des_ev, y=prop, cv=2)
                f_values, p_values = f_regression(X=des_ev, y=prop)
                result = (r_2, f_values, scores, model.intercept_, model.coef_, title)
                res.append(result)
        res = pd.DataFrame(res, columns=["R2", "F", "CV_R2", "Ordenada", "Coef_", "Titulo"])
        self.results = res
        tf = time()
        print("Correlaciones Realizadas:", self.comb.shape[0])
        print("# de propiedades:", prop.shape[0])
        print("Tiempo:", tf - ti)

    def print_results(self, sortby="R2", ascending=False):
        if not len(self.results):
            print(f"No se encontraron correlaciones con R2 mayor a {self.r_ref}")
            return None
        if sortby not in self.results.columns:
            sortby = "R2"
        results = self.results.sort_values(
            by=sortby,
            kind="mergesort",
            ascending=ascending
        )
        results.reset_index(drop=True, inplace=True)
        for result in results.itertuples():
            if result.Index >= self.res_lim:
                break
            print(
                f"{result.Titulo} R2: {result.R2:.3f} Cv_R2: {result.CV_R2[0]:.3f}",
                f"Ordenada: {result.Ordenada:.3f} Coef: {result.Coef_}",
                f"F: {result.F}\n"
            )

    @staticmethod
    def _ask_restrictions():
        conditions = [[0, 0, 0]]
        print("Introduce tu condición:\n Formato: Columna,condición,operador lógico \n Operadores Lógicos: \n &: (y), \n | (o) \n ~ (no) \n Para terminar operador lógico = -1")
        while conditions[-1][2] != "-1":
            cond_v = input()
            conditions.append(cond_v.split(","))
        fcond = ""
        conditions.pop(0)                            # Deleting the first item is not great -> O(n)
        for condition in conditions:
            fcond += f"(file[{condition[0]}]{condition[1]})"
            fcond = f"{fcond} {condition[2]}" if condition[2] != "-1" else fcond
        return fcond


def parse_args():
    ap = ar.ArgumentParser()
    ap.add_argument("FILE", help="El archivo de entrada")
    ap.add_argument("-s", "--sep", dest="SEP", type=str, default=",", required=False, help="separador en el CSV")
    return ap.parse_args()


def main(file, sep):
    mlr = Corr(file)
    mlr.read_csv(sep)
    mlr.set_params()
    mlr.corr()
    mlr.print_results()


if __name__ == "__main__":
    args = parse_args()
    main(file=args.FILE, sep=args.SEP)
