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
            self.res_lim = int(input("Número máximo de resultados a mostrar:"))
        else:
            self.res_lim = 0
        np.set_printoptions(precision=3)
        ti = time()
        prop = self.datos[:, self.num]
        res = []
        for i in range(0, self.comb.shape[0]):
            if self.num_des != 0:
                des_ev = self.datos[:, self.comb[i]]
            else:
                des_ev = self.datos[:, self.comb[i][1]]
                prop = self.datos[:, self.comb[i][0]]
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
            for j in range(0, self.comb.shape[1]):
                if self.n == 0:
                    title = title + " " + str(self.noms[self.comb[i][j]])
                else:
                    title = title + " " + \
                        str(self.noms[self.comb[i][j]]) + "**" + str(self.n)
            # print(des_ev[0])
            # print(prop[0])
            model = LinearRegression().fit(des_ev, prop)
            r_2 = model.score(des_ev, prop)
            if r_2 >= self.r_ref:
                scores = cross_val_score(model, des_ev, prop, cv=2)
                f, p = f_regression(des_ev, prop)
                if self.res_lim != 0:
                    a = (r_2, f, scores, model.intercept_, model.coef_, title)
                    res.append(a)
                else:
                    print(title, "R2:", "{:.3f}".format(r_2), "Cv_R2:", "{:.3f}".format(scores[0]), "Ordenada:", "{:.3f}".format(
                        model.intercept_), "Coef:", model.coef_, "F:", f, "\n")
        dtype = [("R2", np.float), ("F", np.ndarray), ("CV_R2", np.ndarray),
                 ("Ordenada", np.float), ("Coef_", np.ndarray), ("Titulo", "S100")]
        res = np.array(res, dtype=dtype)
        res = np.sort(res, kind="stable", order="R2")
        if self.res_lim != 0 and res.shape[0] > self.res_lim:
            for i in range(res.shape[0] - self.res_lim, res.shape[0]):
                print(str(res[i]["Titulo"]), " R2:", "{:.3f}".format(
                    res[i]["R2"]), "Cv_R2:", "{:.3f}".format(res[i]["CV_R2"][0]), " Ordenada:", "{:.3f}".format(res[i]["Ordenada"]), " Coef:", res[i]["Coef_"], " F:", res[i]["F"], "\n")
        elif self.res_lim != 0 and res.shape[0] < self.res_lim:
            for i in range(0, res.shape[0]):
                print(str(res[i]["Titulo"]), " R2:", "{:.3f}".format(
                    res[i]["R2"]), " Ordenada:", "{:.3f}".format(res[i]["Ordenada"]), " Coef:", res[i]["Coef_"], " F:", res[i]["F"], "\n")
        tf = time()
        print("Correlaciones Realizadas:", self.comb.shape[0])
        print("# de propiedades:", prop.shape[0])
        print("Tiempo:", tf - ti)

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


if __name__ == "__main__":
    args = parse_args()
    main(file=args.FILE, sep=args.SEP)
