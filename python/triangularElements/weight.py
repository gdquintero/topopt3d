from numpy import floor, zeros, pi, exp
def weight(struct, rmin, numNei, distNei):
    rminf = int(floor(rmin))
    # ncol = (2*rminf + 1)**2
    ncol = len(distNei)
    print(ncol)
    weigh = zeros((int(struct["nelem"]), 2*ncol))
    wi = zeros(int(struct["nelem"]))
    rmin32 = 2*((rmin/3)**2)
    twopirmin3 = 2*pi*rmin/3
    for k in range(1, int(struct["nelem" ]) + 1):
        nei = numNei[k - 1] 
        for j in range(1, nei + 1): 
            weigh[k-1,j-1] = exp(-((distNei[j-1,k-1])**2)/rmin32)/twopirmin3
        wi[k - 1] = sum(weigh[k - 1, 0:nei])
    return weigh, wi