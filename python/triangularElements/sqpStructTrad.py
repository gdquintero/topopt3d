# --- Librería estándar ---
import warnings as wn
from time import time
from pyinstrument import Profiler
# --- Librerías de terceros ---
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from numpy import (
    ravel, ones, arange, meshgrid, array, zeros, double, identity, diag
)
# from scipy.sparse import identity, diags
from scipy.sparse.linalg import cg
from scipy.optimize import minimize

# --- Módulos locales / propios del proyecto ---
from elem_stiff import elem_stiff
from get_pos import get_pos
# from get_rows_cols import get_rows_cols
# from global_stiff import global_stiff
from global_stiff_trad import global_stiff_trad
from put_supports import put_supports
from vetfneq import vetfneq
from get_neighborhoodTrad import get_neighborhoodTrad
# from get_neighborhood import get_neighborhood
from weight import weight
from mean_density_filter import mean_density_filter
from graddens import graddens
from grad_compliance_filter import grad_compliance_filter
from grad_compliance import grad_compliance
from sqpSolver import sqp_diagonal
wn.filterwarnings("ignore")
# def obj(x):


def sqpStruct(struct,density,minDens,penal, volfrac, rmin):
    #Ensamblamos matriz de rigidez global
    nelem = int(struct["nelem"])
    ny = int(struct["nnodesy"])
    nx = int(struct["nnodesx"])
    kOdd, kEven = elem_stiff(struct)
    b = float(struct["b"])
    h = float(struct["h"])
    e = float(struct["e"])
    # print(kOdd, kEven)
    # row, col, inic = get_rows_cols(struct)
    # pos = get_pos(struct, inic)
    # row = row - 1
    # col = col - 1
    # pos = pos - 1
    # kOddc = ravel(kOdd)
    # kEvenc = ravel(kEven)
    volS = b*h*e
    vElem = volS / nelem
    vElemArray = vElem * ones(int(nelem))
    volMax = volfrac*volS

    # volfrac = velem/volS

    f = vetfneq(struct)
    supp = put_supports(struct)
    supp = supp - 1
    numNei, neighbsEl, distnei = get_neighborhoodTrad(struct, rmin)
    weigh, wi = weight(struct, rmin, numNei, distnei)
    gradxnew = graddens(struct, weigh, wi, numNei, neighbsEl)
    ap = identity(2*int(struct["nodesNumber"]), dtype =double)
    

    u0 = zeros(2*int(struct["nodesNumber"]))
    def obj(x):
        nonlocal u0
        xvol = mean_density_filter(struct, x, weigh, wi, numNei, neighbsEl)

        start = time()
        K = global_stiff_trad(struct,  kEven, kOdd, xvol, penal)
        end = time()
        print(f"Rigidez : {end - start}")


        K[supp, :] = ap[supp, :]
        K[:, supp] = ap[:, supp]
        M = diag(1.0 / diag(K))
        start = time()
        u, info = cg(K, f , M = M, rtol = 1e-3, maxiter = 2*int(struct["nodesNumber"]), x0 = u0 )
        u0 = u
        end = time()
        print(f"sistema: {end - start}" )
        # print(u, info)

        start = time()
        gradxdens = grad_compliance(struct, u, penal, xvol, kEven, kOdd)
        compliance = grad_compliance_filter(struct, gradxdens, gradxnew, numNei, neighbsEl)
        end = time()
        print(f"compliance {end - start}" )

        return f @ u, compliance
    def ineqConstrain(x):

        #weigh, wi = weight(struct, rmin, numNei, distnei)
        xvol = mean_density_filter(struct, x, weigh, wi, numNei, neighbsEl)
        val = 1 - vElemArray @ xvol / volMax


        return val
    
    def gradIneqConstrain(x):
        gradVol = zeros(nelem)
        #weigh, wi = weight(struct, rmin, numNei, distnei)
        for i in range(1, nelem + 1):
            nei = numNei[i - 1]
            ind = neighbsEl[0:nei, i- 1]
            gradVol[i - 1] = -sum(weigh[i-1, 0:nei] * vElemArray[ind - 1]/wi[ind - 1])/volMax

        return gradVol
 
    constraints = {
        'type': 'ineq',
        'fun': ineqConstrain,
        'jac': gradIneqConstrain  
    }


    lows = minDens*ones(int(nelem))
    uppers = ones(int(nelem))
    bounds = list(zip(lows, uppers))
    # bounds = Bounds(lb=full(struct.nelem, minDens), ub=ones(struct.nelem))

    # print(bounds)
    # def make_callback():
    #     iter_count = [0]
    
        # def callback(x):
        #     iter_count[0] += 1
        #     print(f"Iter {iter_count[0]}")
    
        # return callback
    
    result = minimize(obj , 
                      density, 
                      method=sqp_diagonal, 
                      constraints= constraints, 
                      bounds=bounds, 
                      options = {'maxiter': 400, 'disp' : True, "warm_start": True, "qp_solver" :"osqp"},
                      tol= 1e-3,
                      jac = True,) 
                    #   callback=make_callback())
    # print(result, result.x)
    
    # print(volMax)

    # print(result.x @ vElemArray)
    densityStar = result.x

    x = arange(nx)
    y = arange(ny)

    X, Y = meshgrid(x, y)

    xv = X.flatten()
    yv = Y.flatten()

    triangles = []

    for i in range(nx - 1):
        for j in range(ny - 1):

            p0 = j * nx + i
            p1 = p0 + 1
            p2 = p0 + nx
            p3 = p2 + 1

            triangles.append([p0, p1, p3])  
            triangles.append([p0, p3, p2])  

    triangles = array(triangles)
    triang = tri.Triangulation(xv, yv, triangles)
    plt.figure(figsize=(6,6))

    plt.tripcolor(
        triang,
        facecolors=densityStar,
        edgecolors='black',
        cmap='gray_r',
        vmin=0.001,
        vmax=0.6
    )   

    # plt.colorbar()
    plt.show()

    


    return densityStar


    
struct = pd.read_json("./python/triangularElements/Candelabro.json")
# print(struct)

# print(plt.colormaps)
density = 0.2*ones(int(struct["nelem"]), dtype=double)
minDens = 0.001
penal = 3
frmax = 0.5
# print(plt.colormaps)
# import pstats
# from io import StringIO

# Datos de prueba para llamar la función
# (ajusta con tus valores reales)
# pr = cProfile.Profile()
# pr.enable()

# u0 = None
profiler = Profiler()
profiler.start()
result = sqpStruct(struct, density, minDens, penal, frmax, 2.5)
profiler.stop()
profiler.open_in_browser()
# print(sqpStruct(struct, 0.2*ones(int(struct["nelem"])), 0.001, 3, 0.25, 3.5))


