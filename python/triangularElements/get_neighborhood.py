# import pandas as pd
# from weight import weight
from numpy import floor, zeros, sqrt
# struct = pd.read_json("./python/triangularELements/str3x3.json")

def get_neighborhood(struct, rmin):
    rminf = int(floor(rmin))
    nelem = int(struct["nelem"])
    nrow = (2*rminf + 1)**2

    neighbsEl = zeros((2*nrow, nelem), dtype=int)
    numNei = zeros(nelem, dtype=int)
    nelemx = int(struct["nelemx"])
    nelemy = int(struct["nelemy"])
    numSquarex = int(nelemx/2)
    numSquarey = int(nelemy/2)
    hx = struct["b"]/numSquarex
    hy = struct["h"]/numSquarey

    distnei = zeros((2*nrow, nelem))
    for i in range(1, numSquarex + 1):
        for j in range(1, numSquarey + 1):
            m1 = (i - 1)*numSquarey + j - 1
            ind = 0
            kmin = max(i-rminf,1)
            kmax = min(i+rminf,numSquarex)
            lmin = max(j-rminf,1)
            lmax = min(j+rminf,numSquarey)
            for k in range(kmin, kmax + 1):
                for l in range(lmin, lmax + 1):
                    m2 = (k - 1)*numSquarey + l - 1
                    m2Inf = 2*m2

                    # m2Sup = 2*m2 + 1
                    neighbsEl[2*ind, 2*m1], neighbsEl[2*ind, 2*m1 + 1] = m2Inf + 1, m2Inf + 1
                    neighbsEl[2*ind + 1, 2*m1], neighbsEl[2*ind + 1, 2*m1 + 1] = neighbsEl[2*ind, 2*m1] + 1, neighbsEl[2*ind, 2*m1 + 1] + 1

                    Cinf = [(i - 1/3)*hx, (j - 2/3)*hy]
                    Csup = [(i - 2/3)*hx, (j - 1/3)*hy]
                    CinfNei = [(k - 1/3)*hx, (l - 2/3)*hy]
                    CsupNei = [(k - 2/3)*hx, (l - 1/3)*hy]

                    distnei[2*ind, 2*m1]= sqrt((Cinf[0] - CinfNei[0])**2 + (Cinf[1] - CinfNei[1])**2)
                    distnei[2*ind + 1, 2*m1] = sqrt((Cinf[0] - CsupNei[0])**2 + (Cinf[1] - CsupNei[1])**2)

                    distnei[2*ind, 2*m1 + 1] = sqrt((Csup[0] - CinfNei[0])**2 + (Csup[1] - CinfNei[1])**2)
                    distnei[2*ind + 1, 2*m1 + 1] = sqrt((Csup[0] - CsupNei[0])**2 + (Csup[1] - CsupNei[1])**2)

                    # distnei[ind, m1] = sqrt(((i-k))**2 + ((j-l))**2)
                    ind = ind + 1
            numNei[2*m1] = 2*ind
            numNei[2*m1 + 1] = numNei[2*m1] 
    # weigh, wi = weight(struct, rmin, numNei, distnei)
    return numNei, neighbsEl, distnei

# numNei, neighbsEl, distnei = get_neighborhood(struct, 1)
# # print("")
# print(neighbsEl)
# print(" ")
# print(numNei)
# print("")
# print(distnei)
# print(weigh)
# print("")
# print(wi)