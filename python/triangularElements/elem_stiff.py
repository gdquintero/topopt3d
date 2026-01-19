from numpy import zeros
from numpy import array

def getABC(X, Y, A):
    a = zeros(3)
    b = zeros(3)
    
    a[0] = 1/(2*A) *(X[1]*Y[2] - X[2]*Y[1])
    a[1] = 1/(2*A) *(X[2]*Y[0] - X[0]*Y[2])
    a[2] = 1/(2*A) *(X[0]*Y[1] - X[1]*Y[0])

    b[0] = 1/(2*A)*(Y[1] - Y[2])
    b[1] = 1/(2*A)*(Y[2] - Y[0])
    b[2] = 1/(2*A)*(Y[0] - Y[1])


    return (a, b)

def getK(a, b, A, E, v, tck):
    a1, a2, a3 = a
    b1, b2, b3 = b

    d1 = 2*v*v - 2
    d2 = 2*v - 2

    
    K11 = (-2*a1*a1*A*E*tck - b1*b1*A*E*tck + b1*b1*v*A*E*tck)/d1
    K22 = (-a1*a1*A*E*tck - 2*b1*b1*A*E*tck + a1*a1*v*A*E*tck)/d1
    K33 = (-2*a2*a2*A*E*tck - b2*b2*A*E*tck + b2*b2*v*A*E*tck)/d1
    K44 = (-a2*a2*A*E*tck - 2*b2*b2*A*E*tck + a2*a2*v*A*E*tck)/d1
    K55 = (-2*a3*a3*A*E*tck - b3*b3*A*E*tck + b3*b3*v*A*E*tck)/d1
    K66 = (-a3*a3*A*E*tck - 2*b3*b3*A*E*tck + a3*a3*v*A*E*tck)/d1

    
    K12 = (-a1*b1*A*E*tck)/d2
    K13 = (-2*a1*a2*A*E*tck - b1*b2*A*E*tck + b1*b2*v*A*E*tck)/d1
    K14 = (-a2*b1*A*E*tck + a2*b1*v*A*E*tck - 2*a1*b2*A*E*tck)/d1
    K15 = (-2*a1*a3*A*E*tck - b1*b3*A*E*tck + b1*b3*v*A*E*tck)/d1
    K16 = (-a3*b1*A*E*tck + a3*b1*v*A*E*tck - 2*a1*b3*A*E*tck)/d1

    K23 = (-a1*b2*A*E*tck - 2*a2*b1*A*E*tck + a1*b2*v*A*E*tck)/d1
    K24 = (-a1*a2*A*E*tck - 2*b1*b2*A*E*tck + a1*a2*v*A*E*tck)/d1
    K25 = (-a1*b3*A*E*tck - 2*a3*b1*A*E*tck + a1*b3*v*A*E*tck)/d1
    K26 = (-a1*a3*A*E*tck - 2*b1*b3*A*E*tck + a1*a3*v*A*E*tck)/d1

    K34 = (-a2*b2*A*E*tck)/d2
    K35 = (-2*a2*a3*A*E*tck - b2*b3*A*E*tck + b2*b3*v*A*E*tck)/d1
    K36 = (-a3*b2*A*E*tck + a3*b2*v*A*E*tck - 2*a2*b3*A*E*tck)/d1

    K45 = (-a2*b3*A*E*tck - 2*a3*b2*A*E*tck + a2*b3*v*A*E*tck)/d1
    K46 = (-a2*a3*A*E*tck - 2*b2*b3*A*E*tck + a2*a3*v*A*E*tck)/d1

    K56 = (-a3*b3*A*E*tck)/d2

    
    K = array([
        [K11, K12, K13, K14, K15, K16],
        [K12, K22, K23, K24, K25, K26],
        [K13, K23, K33, K34, K35, K36],
        [K14, K24, K34, K44, K45, K46],
        [K15, K25, K35, K45, K55, K56],
        [K16, K26, K36, K46, K56, K66],
    ])
    
    return K
    

def elem_stiff(struct):
    
    h = int(struct.h)
    b = int(struct.b)
    v = float(struct.v)
    E = float(struct.E)
    nelemx = int(struct.nelemx)
    nelemy = int(struct.nelemy)
    tck = float(struct.e)

    x1, y1 = (0, 0)
    x2, y2 = (2*b/nelemx, 0)
    x3, y3 = (2*b/nelemx, 2*h/nelemy)
    x4, y4 = (0, 2*h/nelemy)
    Aeven = 1/2 * (x2*y3 - x3*y2 + y2*x1 - y3*x1 + x3*y1 - x2*y1) 

    (aOdd, bOdd), (aEven, bEven) = getABC([x1, x2, x3], [y1, y2, y3], Aeven), getABC([x1, x3, x4], [y1, y3, y4], Aeven)
    Kodd = getK(aOdd, bOdd, Aeven, E, v, tck)
    Keven = getK(aEven, bEven, Aeven, E, v, tck)

    return (Kodd, Keven)

