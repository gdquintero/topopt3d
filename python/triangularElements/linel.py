from numpy import zeros

def linel(struct):
    nelemy = struct["nelemy"][0]
    nelem = int(struct["nelem"])
    nsquare = int(nelem/2)
    lin = zeros((6, nelem), dtype= int)

    for i in range(1, nsquare+1):
        node = i + int(i/(nelemy/2)) if i%(nelemy/2) != 0 else i + int(i/(nelemy/2))- 1 
        lin[1, 2*i - 1 ], lin[1, 2*i - 2] =  2*node, 2*node
        lin[0, 2*i - 1 ], lin[0, 2*i - 2] =  lin[1, 2*i - 1 ] - 1, lin[1, 2*i - 1 ] - 1

        lin[3, 2*i - 2] = lin[1, 2*i - 2] + 2*(nelemy/2 + 1)
        lin[2, 2*i - 2] = lin[3, 2*i - 2] - 1

        lin[3, 2*i - 1], lin[5, 2*i - 2] = lin[1, 2*i - 2] + 2 + 2*(nelemy/2 + 1), lin[1, 2*i - 2] + 2 + 2*(nelemy/2 + 1)
        lin[2, 2*i - 1],  lin[4, 2*i - 2] = lin[3, 2*i - 1] - 1, lin[3, 2*i - 1] - 1

        lin[5, 2*i - 1] = lin[1, 2*i - 2] + 2
        lin[4, 2*i - 1] = lin[5, 2*i - 1] - 1
        
    return lin
