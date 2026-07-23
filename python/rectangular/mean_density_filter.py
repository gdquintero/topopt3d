from numpy import zeros
def mean_density_filter(struct,x,weigh,wi,numNei,neigh):
    nelem = int(struct["nelem"])
    xvol = zeros(nelem)
    for i in range(1, nelem + 1):
        nei = numNei[i - 1]
        ind = neigh[0:nei, i - 1]
        xvol[i - 1] = weigh[i - 1, 0:nei] @ x[ind - 1]/wi[i - 1]
    return xvol