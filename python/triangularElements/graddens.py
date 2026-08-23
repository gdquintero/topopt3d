# import pandas as pd
# from get_neighborhood import get_neighborhood
# from weight import weight
from numpy import ones
from scipy.sparse import csr_matrix
# struct = pd.read_json("./python/rectangular/struct5x4.json")


def graddens(struct, weigh, wi, numNei, elnei):
    row = []
    col = []
    data = []
    for i in range(1, int(struct["nelem"]) + 1):
        nei = numNei[i - 1]
        ind = elnei[0:nei, i -1]
        val = weigh[i - 1, 0:nei]/wi[ind - 1]
        col = [*col, *(ind - 1)]
        row = [*row, *((i-1)*ones(len(ind)))]
        data = [*data, *val]


    gradxnew = csr_matrix((data, (row, col)), shape=(int(struct["nelem"]), int(struct["nelem"])))
    return gradxnew
# numNei, elnei, distnei = get_neighborhood(struct, 1)
# print(distnei)
# print("")
# weigh, wi = weight(struct, 1, numNei, distnei)
# print(weigh)
# print("")
# print(wi)
# print(graddens(struct, weigh, wi, numNei, elnei))