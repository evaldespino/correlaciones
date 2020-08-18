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

    def rdcvs(self, sep):
        print("Leyendo los datos del CSV")
        file = pd.read_csv(self.path, sep)
        print("Se encontraron", file.shape[1], "Descriptores y", file.shape[0], "datos")
        print("¿Restringir base de datos? \n 1: Sí \n 2: No ")
        res = int(input())
        if res == 1:
            cond = [[0, 0, 0]]
            print("Introduce tu condición:\n Formato: Columna,condición,operador lógico \n Operadores Lógicos: \n &: (y), \n | (o) \n ~ (no) \n Para terminar operador lógico = -1")
            while cond[len(cond) - 1][2] != "-1":
                cond_v = input()
                user_cond = cond_v.split(",")
                cond.append(user_cond)
            fcond = ""
            cond.pop(0)
            for i in range(0, len(cond)):
                if cond[i][2] != "-1":
                    fcond = fcond + "(file[" + cond[i][0] + "]" + \
                        cond[i][1] + ")" + " " + cond[i][2]
                else:
                    fcond = fcond + " (file[" + cond[i][0] + "]" + cond[i][1] + ")"
            file = file[eval(fcond)]
        file = file.select_dtypes(include=np.number)
        self.noms = []
        for col_name in file:
            self.noms.append(col_name)
        self.datos = file.to_numpy()
        print("Se encontraron", len(self.noms),
              "Descriptores numéricos y", self.datos.shape[0], "datos")

    def param(self):
        print("# Columna a Evaluar (0 - ", len(self.noms) - 1, ")")
        for i in range(0, len(self.noms)):
            print(str(i) + ":", self.noms[i])
        self.num = int(input())
        print(self.noms[self.num], "será la propiedad a evaluar")
        print("Escoge la Rutina:\n 1: Correlación entre Descriptores \n 2: Correlación De propiedades \n 3: Correlaciones polinomiales \n 4: Correlaciones con Descriptores a la n")
        ent = int(input())
        print("R2 mínima")
        self.r_ref = float(input())
        indexes = np.array([i for i in range(0, self.datos.shape[1])])
        indexes = indexes[(indexes != self.num)]
        if ent == 2:
            print("# Descriptores")
            self.num_des = int(input())
            self.comb = np.array(list(it.combinations(indexes, self.num_des)))
            self.or_pol = 0
            self.n = 0
            print(self.comb.shape[0])
        elif ent == 1:
            self.comb = np.array(list(it.combinations(indexes, 2)))
            self.num_des = 0
            self.or_pol = 0
            self.n = 0
        elif ent == 3:
            print("Orden máxmimo del polinomio")
            self.or_pol = int(input())
            self.comb = indexes.reshape((-1, 1))
            self.num_des = -1
            self.n = 0
        elif ent == 4:
            print("Escribe n (des**n):")
            self.n = float(input())
            self.or_pol = 0
            self.num_des = -1
            self.comb = indexes.reshape((-1, 1))

    def corr(self):
        print("Preprocesamiento:\n 1:Ninguno \n 2:Normalizar \n 3:Escalar")
        ent_pre = int(input())
        print("Limitar Resultados:\n 1:Sí \n 2:No")
        ent_lim = int(input())
        if ent_lim == 1:
            print("# de Resultados:")
            self.res_lim = int(input())
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
            if self.or_pol != 0:
                des_ev = PolynomialFeatures(
                    degree=self.or_pol, include_bias=False).fit_transform(des_ev)
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


def parse_args():
    ap = ar.ArgumentParser()
    ap.add_argument("FILE", help="El archivo de entrada")
    ap.add_argument("-s", "--sep", dest="SEP", type=str, default=",", required=False, help="separador en el CSV")
    return ap.parse_args()


def main(file, sep):
    mlr = Corr(file)
    mlr.rdcvs(sep)
    mlr.param()
    mlr.corr()


if __name__ == "__main__":
    args = parse_args()
    main(file=args.FILE, sep=args.SEP)
