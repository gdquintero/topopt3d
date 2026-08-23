# --- Librerías estándar ---
import time
import warnings as wn
import cProfile
from pyinstrument import Profiler
# --- Librerías de terceros ---
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ravel, ones, double, reshape, zeros, diag
# from scipy.sparse import identity, diags
from numpy import identity
from scipy.sparse.linalg import cg, LinearOperator, spilu
# from scipy.sparse import diags
from scipy.optimize import minimize
from sqpSolver import sqp_diagonal as diagonal_sqp
# --- Módulos propios ---
from elem_stiff import elem_stiff
from get_pos import get_pos
from get_rows_cols_na import get_rows_cols
from global_stiff_tradv2 import global_stiff_tradv2
from put_supports import put_supports
from vetfneq import vetfneq
from grad_compliance import grad_compliance
from get_neighborhood import get_neighborhood
from weight import weight
from mean_density_filter import mean_density_filter
from graddens import graddens
from grad_compliance_filter import grad_compliance_filter
wn.filterwarnings("ignore")


def sqpStruct(struct,density,minDens,penal, volfrac, rmin):
    #Ensamblamos matriz de rigidez global
    
    kel = elem_stiff(struct)
    b = float(struct["b"])
    h = float(struct["h"])
    e = float(struct["e"])
    nelem = int(struct["nelem"])

    row, col, inic = get_rows_cols(struct)
    pos = get_pos(struct, inic)
    # row = row - 1
    # col = col - 1
    # pos = pos - 1
    kelc = kel
    volS = b*h*e
    vElem = volS / nelem
    vElemArray = vElem * ones(nelem)
    volMax = volfrac*volS

    # print(vElemArray)

    # volfrac = velem/volS

    f = vetfneq(struct)
    supp = put_supports(struct)
    supp = supp - 1
    numNei, neighbsEl, distnei = get_neighborhood(struct, rmin)
    weigh, wi = weight(struct, rmin, numNei, distnei)
    gradxnew = graddens(struct, weigh, wi, numNei, neighbsEl)
    ap = identity(2*int(struct["nodesNumber"]), dtype = double)

    obj_time = 0.0
    obj_calls = 0
    u0 = zeros(2*int(struct["nodesNumber"]))
    def obj(x):
        nonlocal obj_time, obj_calls, u0
        # print(u0)
        t0 = time.perf_counter()
        xvol = mean_density_filter(struct, x, weigh, wi, numNei, neighbsEl)
        # xvol = x
        #x = xvol
        start = time.time()
        # global_stiff_tradv2()
        K = global_stiff_tradv2(struct, kel, xvol, penal)
        end = time.time()
        print(f"rigidez: {end - start}")

        
        # start = time.time()
        K[supp, :] = ap[supp, :]
        K[:, supp] = ap[:, supp]
        # D = diags(K.diagonal())
        # end = time.time()
        # print(end - start)

        # print(K)
        # K = K.astype(float32)
        
        # ilu = spilu(K.tocsc(), fill_factor=10, drop_tol=1e-4)
        # diago = diag(K)
        # M = LinearOperator(
        #     K.shape,
        #     matvec=lambda x: x / diago
        # )

        M = diag(1.0 / diag(K))
        # print(M)
        # M = diags(K.diagonal()).tocsr()
        start = time.time()
        u, info = cg(K, f, M= M, rtol = 1e-3, maxiter = 2*int(struct["nodesNumber"]), x0 = u0)
        u0 = u
        end = time.time()
        print(f"sistema: {end - start}")

        # print(u, info)
        # u0 = u.copy()

        #compliance = grad_compliance(struct, u, penal, xvol, kel) 
        start = time.time()
        gradxdens = grad_compliance(struct, u, penal, xvol, kel)
        compliance = grad_compliance_filter(struct, gradxdens, gradxnew, numNei, neighbsEl)
        end = time.time()
        print(f"compliance : {end - start}")
        obj_time += time.perf_counter() - t0
        obj_calls += 1
        # print(grad_compliance_filter(struct, gradxdens, gradxnew, numNei, neighbsEl))

        return f @ u, compliance
    
    con_time = 0.0
    con_calls = 0
    def ineqConstrain(x):
        nonlocal con_time, con_calls

        t0 = time.perf_counter()
        #weigh, wi = weight(struct, rmin, numNei, distnei)
        xvol = mean_density_filter(struct, x, weigh, wi, numNei, neighbsEl)
        val = 1 - vElemArray @ xvol / volMax

        con_time += time.perf_counter() - t0
        con_calls += 1

        return val
    jac_time = 0.0
    jac_calls = 0
    def gradIneqConstrain(x):
        nonlocal jac_time, jac_calls
        t0 = time.perf_counter()
        gradVol = zeros(nelem)
        #weigh, wi = weight(struct, rmin, numNei, distnei)
        for i in range(1, nelem + 1):
            nei = numNei[i - 1]
            ind = neighbsEl[0:nei, i- 1]
            gradVol[i - 1] = -sum(weigh[i-1, 0:nei] * vElemArray[ind - 1]/wi[ind - 1])/volMax

        jac_time += time.perf_counter() - t0
        jac_calls += 1
        return gradVol
 
    constraints = {
        'type': 'ineq',
        'fun': ineqConstrain,
        'jac': gradIneqConstrain  
    }

    lows = minDens*ones(nelem)
    uppers = ones(nelem)
    bounds = list(zip(lows, uppers))
    # bounds = Bounds(lb=full(struct.nelem, minDens), ub=ones(struct.nelem))

    # print(bounds)

    def make_callback():
        iter_count = [0]
    
        def callback(x):
            iter_count[0] += 1
            print(f"Iter {iter_count[0]}")
    
        return callback

    start = time.perf_counter()
    result = minimize(obj , 
                      density, 
                      method= diagonal_sqp, 
                      constraints= constraints, 
                      bounds=bounds,
                      options = {'maxiter': 400, 'disp' : True, "warm_start": True,  "qp_solver" :"osqp"},
                    #   verbose = True,
                      tol= 1e-3,
                      jac= True,)
                    #   callback= make_callback())

    end = time.perf_counter()

    print("Tiempo total:", end - start)
    densityStar = result["x"]
    # print(result.nit)
    # print(result.nfev)
    # print(result.njev)
    # print(result.message)
    # print(result.success)
    # print(result)
    R = -reshape(densityStar, (int(struct["nelemx"]), int(struct["nelemy"]))).T
    
    for i in range(R.shape[0] + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)

    for j in range(R.shape[1] + 1):
        plt.axvline(j - 0.5, color='black', linewidth=0.5)
        
    plt.set_cmap('gray')
    plt.imshow(R, origin= "lower", vmax = -0.001)
    plt.colorbar()
    plt.show()
    print("obj:", obj_calls, obj_time)
    print("con:", con_calls, con_time)
    print("jac:", jac_calls, jac_time)

    return densityStar


    # # print(f)
    # f0val = f @ u

    # print(u)
    # print(f0val)
    
struct = pd.read_json("./python/rectangular/struct3x3.json")

# struct = {
#     "nelemx" : 120,
#     "nelemy" : 30,
#     "nnodesx" : 121,
#     "nnodesy" : 31,
#     "nelem" : 3600,
#     "h" : 30,
#     "b" : 120,    
#     "E" : 3000,
#     "v" : 0.3,
#     "e" : 1,

#     "forces" : 
#         [{
#             "Fx" : [0],
#             "Fy" : [-120], 
#             "node" : [3736]
#         }]
#     ,
#     "supp" : 
#         [{
#             "x" : [0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0],
#             "y" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
#             "node" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
#             "ix" : [1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1, 1, 1 ,1, 1 , 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1, 1, 1 ,1, 1],
#             "iy" : [1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1, 1, 1 ,1, 1 , 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1 ,1, 1, 1, 1 ,1, 1]
#         }],

#     "nodesNumber": 3751

# }
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
result = sqpStruct(struct, density, minDens, penal, frmax, 2)
profiler.stop()
profiler.open_in_browser()
# pr.disable()

# Imprimir resultados ordenados por tiempo acumulado
# s = StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(20)  # top 20 funciones más costosas
# ps.print_stats("global_stiff")
# ps.print_stats("cg")
# ps.print_stats("grad_compliance")
# ps.print_stats("grad_compliance_filter")
# ps.print_stats("mean_density")

# print(s.getvalue())


