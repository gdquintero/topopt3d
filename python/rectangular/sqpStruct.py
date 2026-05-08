from elem_stiff import elem_stiff
from get_pos import get_pos
from get_rows_cols_na import get_rows_cols
from global_stiff import global_stiff
from put_supports import put_supports
from vetfneq import vetfneq
from numpy import ravel
from numpy import ones
from numpy import reshape
from scipy.sparse import identity
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
# from scipy import minimize
import pandas as pd
import warnings as wn
import matplotlib.pyplot as plt

wn.filterwarnings("ignore")


def sqpStruct(struct,density,minDens,penal, volfrac):
    #Ensamblamos matriz de rigidez global
    kel = elem_stiff(struct)[:, :, 0]
    row, col, inic = get_rows_cols(struct)
    pos = get_pos(struct, inic)
    row = row - 1
    col = col - 1
    pos = pos - 1
    kelc = ravel(kel)
    volS = struct.b * struct.h * struct.e
    vElem = volS / struct.nelem
    vElemArray = vElem[0] * ones(int(struct.nelem))
    volMax = volfrac*volS

    # volfrac = velem/volS

    f = vetfneq(struct)
    supp = put_supports(struct)
    supp = supp - 1

    def obj(x):
        K = global_stiff(struct, x, penal, row, col, inic, kelc, pos)

        ap = identity(2*int(struct.nodesNumber)).tocsr()

        K[supp, :] = ap[supp, :]
        K[:, supp] = ap[:, supp]

        u, shape = cg(K, f)

        return f @ u
    
    constraints = {
        'type': 'ineq',
        'fun': lambda x: volMax - vElemArray @ x  # x.mean() <= volfrac
    }

    lows = minDens*ones(int(struct.nelem))
    uppers = ones(int(struct.nelem))
    bounds = list(zip(lows, uppers))
    # bounds = Bounds(lb=full(struct.nelem, minDens), ub=ones(struct.nelem))

    # print(bounds)

    result = minimize(obj , 
                      density, 
                      method='SLSQP', 
                      constraints= constraints, 
                      bounds=bounds)

    densityStar = result.x
    print(result)
    R = -reshape(densityStar, (int(struct.nelemy), int(struct.nelemx))).T
    
    for i in range(R.shape[0] + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)

    for j in range(R.shape[1] + 1):
        plt.axvline(j - 0.5, color='black', linewidth=0.5)
        
    plt.set_cmap('gray')
    plt.imshow(R, origin= "lower")
    plt.colorbar()
    plt.show()


    return densityStar


    # # print(f)
    # f0val = f @ u

    # print(u)
    # print(f0val)
    
struct = pd.read_json("./python/rectangular/struct3x3.json")

# print(plt.colormaps)
print(sqpStruct(struct, 0.4*ones(int(struct.nelem)), 0.001, 3, 0.5))


