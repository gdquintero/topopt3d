from elem_stiff import elem_stiff
from get_pos import get_pos
from get_rows_cols import get_rows_cols
from global_stiff import global_stiff
from put_supports import put_supports
from vetfneq import vetfneq
from numpy import ravel
from numpy import ones
from numpy import arange
from numpy import meshgrid
from numpy import array
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
# from scipy import minimize
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from grad_compliance import grad_compliance
from time import time
wn.filterwarnings("ignore")
# def obj(x):


def sqpStruct(struct,density,minDens,penal, volfrac):
    #Ensamblamos matriz de rigidez global
    ny = int(struct.nnodesy)
    nx = int(struct.nnodesx)
    kOdd, kEven = elem_stiff(struct)

    # print(kOdd, kEven)
    row, col, inic = get_rows_cols(struct)
    pos = get_pos(struct, inic)
    row = row - 1
    col = col - 1
    pos = pos - 1
    kOddc = ravel(kOdd)
    kEvenc = ravel(kEven)
    volS = struct.b * struct.h * struct.e
    vElem = volS / struct.nelem
    vElemArray = vElem[0] * ones(int(struct.nelem))
    volMax = volfrac*volS

    # volfrac = velem/volS

    f = vetfneq(struct)
    supp = put_supports(struct)
    supp = supp - 1



    def obj(x):
        start = time()
        K = global_stiff(struct, x, penal, row, col, inic, kEvenc, kOddc, pos)
        end = time()
        print(end - start)

        ap = identity(2*int(struct.nodesNumber)).tocsr()

        K[supp, :] = ap[supp, :]
        K[:, supp] = ap[:, supp]
        
        start = time()
        u, info = cg(K, f , rtol = 1e-3 )

        end = time()
        print(end - start)
        print(u, info)

        compliance = grad_compliance(struct, u, penal, x, kEven, kOdd)

        return f @ u, compliance
    
    constraints = {
        'type': 'ineq',
        'fun': lambda x: volMax - vElemArray @ x  
    }

    lows = minDens*ones(int(struct.nelem))
    uppers = ones(int(struct.nelem))
    bounds = list(zip(lows, uppers))
    # bounds = Bounds(lb=full(struct.nelem, minDens), ub=ones(struct.nelem))

    # print(bounds)
    def make_callback():
        iter_count = [0]
    
        def callback(x):
            iter_count[0] += 1
            print(f"Iter {iter_count[0]}")
    
        return callback
    
    result = minimize(obj , 
                      density, 
                      method='SLSQP', 
                      constraints= constraints, 
                      bounds=bounds, 
                      options = {"maxiter" : 20},
                      tol= 1e-3,
                      jac = True, 
                      callback=make_callback())
    print(result, result.x)
    
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
        vmin=0,
        vmax=1
    )   

    plt.colorbar()
    plt.show()

    


    return densityStar


    
struct = pd.read_json("./python/triangularElements/struct3x3.json")


# print(plt.colormaps)
print(sqpStruct(struct, 0.4*ones(int(struct.nelem)), 0.001, 1, 0.5))


