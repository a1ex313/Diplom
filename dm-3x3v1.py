import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy import integrate
from scipy import linalg as LA
from scipy.optimize import minimize
from scipy import optimize
from mpmath import *
import time
import multiprocessing

start = time.time()

#Общие параметры
p = 0.0
x0 = 0.0
z0 = 6.0
l = 0.0001
L = 100.0
R = 6.0


#Параметры для фи1
a1 = (0.5)*(math.sqrt(l))
b1 = (0.5)*(math.sqrt(l))
c1 = (0.5)*(math.sqrt(l))

#Параметры для фи21
a2 = 0.2
b2 = 0.2
c2 = 0.2

#Параметры для фи22
a3 = 1.313
b3 = 1.313
c3 = 1.313

def phi1(x, y, z):
    return z*math.exp(-a1*(x-x0)**2)*math.exp(-b1*y**2)*math.exp(-c1*z**2)
def phi21(x, y, z):
    return math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2)
def phi22(x, y, z):
    return math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2)



def phi1_dr2(x, y, z):
    return (2*a1*z*(2*a1*((x0-x)**2)-1)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2))) + (2*b1*z*(2*b1*(y**2)-1)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2))) + (2*c1*z*(2*c1*(z**2)-3)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2)))
def phi21_dr2(x, y, z):
    return (2*a2*(2*a2*(x**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2))) + (2*b2*(2*b2*(y**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2))) + (2*c2*(2*c2*((z0-z)**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2)))
def phi22_dr2(x, y, z):
    return (2*a3*(2*a3*(x**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2))) + (2*b3*(2*b3*(y**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2))) + (2*c3*(2*c3*((z0-z)**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2)))



def phi1_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2))/math.sqrt(x**2+y**2+(z-z0)**2) - p*l*(x**2+y**2+z**2) * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2))/2
    #else:
    #    return 0.0
def phi21_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2))/math.sqrt(x**2+y**2+(z-z0)**2) - p*l*(x**2+y**2+z**2) * (math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2))/2
    #else:
    #    return 0.0
def phi22_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2))/math.sqrt(x**2+y**2+(z-z0)**2) - p*l*(x**2+y**2+z**2) * (math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2))/2
    #else:
    #    return 0.0



def phi1_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta) + z0
    return z*math.exp(-a1*(x-x0)**2)*math.exp(-b1*y**2)*math.exp(-c1*z**2)*J(r, teta)
def phi21_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta) + z0
    return math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2)*J(r, teta)
def phi22_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta) + z0
    return math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2)*J(r, teta)



def phi1_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta) + z0
    return 2 * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2)) - p*l * r *  (
                x ** 2 + y ** 2 + z ** 2) * (
                z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2)) / 2
def phi21_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta)+ z0
    return 2 * (math.exp(-a2 * x ** 2) * math.exp(-b2 * y ** 2) * math.exp(-c2 * (z - z0) ** 2)) - p*l * r *  (
                x ** 2 + y ** 2 + z ** 2) * (math.exp(-a2 * x ** 2) * math.exp(-b2 * y ** 2) * math.exp(-c2 * (z - z0) ** 2)) / 2
def phi22_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.sin(phi)
    z = r*math.cos(teta) + z0
    return 2 * (math.exp(-a3 * x ** 2) * math.exp(-b3 * y ** 2) * math.exp(-c3 * (z - z0) ** 2)) - p*l * r *  (
                x ** 2 + y ** 2 + z ** 2) * (math.exp(-a3 * x ** 2) * math.exp(-b3 * y ** 2) * math.exp(-c3 * (z - z0) ** 2)) / 2


def J(r, teta):
    return r * math.sin(teta)





def integrate_phi00():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi01():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi02():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi10():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi11():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi12():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, 40, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi20():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi21():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi22():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi10_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi20_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi10_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi20_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi10_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi20_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res


if __name__ == '__main__':
    def get_phi(N):
        start1 = time.time()
        pool1 = multiprocessing.Pool(processes=9)

        res00 = pool1.apply_async(integrate_phi00)
        res01 = pool1.apply_async(integrate_phi01)
        res02 = pool1.apply_async(integrate_phi02)

        res10 = pool1.apply_async(integrate_phi10)
        res11 = pool1.apply_async(integrate_phi11)
        res12 = pool1.apply_async(integrate_phi12)

        res20 = pool1.apply_async(integrate_phi20)
        res21 = pool1.apply_async(integrate_phi21)
        res22 = pool1.apply_async(integrate_phi22)



        phi_m = np.zeros(shape=(N, N))

        phi_m[0][0] = res00.get()
        phi_m[0][1] = res01.get()
        phi_m[0][2] = res02.get()

        phi_m[1][0] = res10.get()
        phi_m[1][1] = res11.get()
        phi_m[1][2] = res12.get()

        phi_m[2][0] = res20.get()
        phi_m[2][1] = res21.get()
        phi_m[2][2] = res22.get()

        end1 = time.time()
        print("Время на матрицу phi")
        print(end1 - start1)
        print()

        return phi_m
    def get_Aphi_1(N):
        start2 = time.time()

        pool2 = multiprocessing.Pool(processes=9)

        resA00 = pool2.apply_async(integrate_Aphi00_1)
        resA01 = pool2.apply_async(integrate_Aphi01_1)
        resA02 = pool2.apply_async(integrate_Aphi02_1)

        resA10 = pool2.apply_async(integrate_Aphi10_1)
        resA11 = pool2.apply_async(integrate_Aphi11_1)
        resA12 = pool2.apply_async(integrate_Aphi12_1)

        resA20 = pool2.apply_async(integrate_Aphi20_1)
        resA21 = pool2.apply_async(integrate_Aphi21_1)
        resA22 = pool2.apply_async(integrate_Aphi22_1)



        Aphi_m = np.zeros(shape=(N, N))

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()

        end2 = time.time()
        print("Время на матрицу Aphi_1")
        print(end2 - start2)
        print()

        return Aphi_m
    def get_Aphi_2(N):
        start2 = time.time()

        pool3 = multiprocessing.Pool(processes=9)

        resA00 = pool3.apply_async(integrate_Aphi00_2)
        resA01 = pool3.apply_async(integrate_Aphi01_2)
        resA02 = pool3.apply_async(integrate_Aphi02_2)

        resA10 = pool3.apply_async(integrate_Aphi10_2)
        resA11 = pool3.apply_async(integrate_Aphi11_2)
        resA12 = pool3.apply_async(integrate_Aphi12_2)

        resA20 = pool3.apply_async(integrate_Aphi20_2)
        resA21 = pool3.apply_async(integrate_Aphi21_2)
        resA22 = pool3.apply_async(integrate_Aphi22_2)



        Aphi_m = np.zeros(shape=(N, N))

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()

        end2 = time.time()
        print("Время на матрицу Aphi_2")
        print(end2 - start2)
        print()

        return Aphi_m

    def get_Aphi_3(N):
        start2 = time.time()

        pool3 = multiprocessing.Pool(processes=9)

        resA00 = pool3.apply_async(integrate_Aphi00_3)
        resA01 = pool3.apply_async(integrate_Aphi01_3)
        resA02 = pool3.apply_async(integrate_Aphi02_3)

        resA10 = pool3.apply_async(integrate_Aphi10_3)
        resA11 = pool3.apply_async(integrate_Aphi11_3)
        resA12 = pool3.apply_async(integrate_Aphi12_3)

        resA20 = pool3.apply_async(integrate_Aphi20_3)
        resA21 = pool3.apply_async(integrate_Aphi21_3)
        resA22 = pool3.apply_async(integrate_Aphi22_3)



        Aphi_m = np.zeros(shape=(N, N))

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()

        end2 = time.time()
        print("Время на матрицу Aphi_3")
        print(end2 - start2)
        print()

        return Aphi_m


    phi_m = get_phi(3)
    print("Матрица phi")
    print(phi_m)
    print("\n")

    Aphi_m1 = get_Aphi_1(3)
    print("Матрица Aphi_1")
    print(Aphi_m1)
    print("\n")


    Aphi_m2 = get_Aphi_2(3)
    print("Матрица Aphi_2")
    print(Aphi_m2)
    print("\n")

    Aphi_m3 = get_Aphi_3(3)
    print("Матрица Aphi_3")
    print(Aphi_m3)
    print("\n")


    Aphi_m = Aphi_m1 + Aphi_m2

    print(Aphi_m)

    eigenVal, eigenVectors = LA.eig(Aphi_m, phi_m)
    print("Собственные значения")
    print(eigenVal)
    print("\n")
    print("Собственные векторы")
    print(eigenVectors)

