from numpy import zeros
# from numpy import dot
def grad_compliance_filter(struct,gradxdens,gradxnew,numNei, neiEl):
    df0dx = gradxnew @ gradxdens
    # print(gradxnew)
    # gradxnew = gradxnew.tocsr()
    # for i in range(1, int(struct["nelem"]) + 1):
    #     ind = numNei[i - 1]
    #     nei = neiEl[0:ind, i - 1]
    #     compliance = 0
    #     # for j in (nei - 1):
    #     #     compliance += gradxnew[i - 1, j]*gradxdens[j]
    #     compliance = gradxnew[i - 1, nei - 1].dot(gradxdens[nei - 1])


    #     df0dx[i - 1] =  compliance[0]

    # df0dx = 
    # print(hola == df0dx)
    return df0dx
