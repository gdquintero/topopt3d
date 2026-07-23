from numpy import zeros

def grad_compliance_filter(struct,gradxdens,gradxnew,numNei, neiEl):
    df0dx = zeros(int(struct["nelem"]))
    for i in range(1, int(struct["nelem"]) + 1):
        ind = numNei[i - 1]
        nei = neiEl[0:ind, i - 1]
        compliance = gradxnew[i - 1, nei - 1] @ gradxdens[nei - 1]
        df0dx[i - 1] =  compliance[0]
    return df0dx
