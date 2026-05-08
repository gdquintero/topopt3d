from numpy import zeros
from numpy import array
def getABC(X, Y, A):
    a = zeros(3)
    b = zeros(3)
    
    b[0] = -1/(2*A)*(X[1] - X[2])
    b[1] = -1/(2*A)*(X[2] - X[0])
    b[2] = -1/(2*A)*(X[0] - X[1])


    a[0] = 1/(2*A)*(Y[1] - Y[2])
    a[1] = 1/(2*A)*(Y[2] - Y[0])
    a[2] = 1/(2*A)*(Y[0] - Y[1])


    return (a, b)

def getK(a, b, A, E, v, tck):
    a1, a2, a3 = a
    b1, b2, b3 = b

    d1 = 2*v*v - 2
    d2 = 2*v - 2


    
    K11 = (-2*A*E*a1**2*tck + A*E*b1**2*tck*v - A*E*b1**2*tck)/d1
    K22 = (A*E*a1**2*tck*v - A*E*a1**2*tck - 2*A*E*b1**2*tck)/d1
    K33 = (-2*A*E*a2**2*tck + A*E*b2**2*tck*v - A*E*b2**2*tck)/d1
    K44 = (A*E*a2**2*tck*v - A*E*a2**2*tck - 2*A*E*b2**2*tck)/d1
    K55 = (-2*A*E*a3**2*tck + A*E*b3**2*tck*v - A*E*b3**2*tck)/d1
    K66 = (A*E*a3**2*tck*v - A*E*a3**2*tck - 2*A*E*b3**2*tck)/d1

    K12 = (-A*E*a1*b1*tck)/d2
    K13 = (-2*A*E*a1*a2*tck + A*E*b1*b2*tck*v - A*E*b1*b2*tck)/d1
    K14 = (-2*A*E*a1*b2*tck*v + A*E*a2*b1*tck*v - A*E*a2*b1*tck)/d1
    K15 = (-2*A*E*a1*a3*tck + A*E*b1*b3*tck*v - A*E*b1*b3*tck)/d1
    K16 = (-2*A*E*a1*b3*tck*v + A*E*a3*b1*tck*v - A*E*a3*b1*tck)/d1

    K23 = (A*E*a1*b2*tck*v - A*E*a1*b2*tck - 2*A*E*a2*b1*tck*v)/d1
    K24 = (A*E*a1*a2*tck*v - A*E*a1*a2*tck - 2*A*E*b1*b2*tck)/d1
    K25 = (A*E*a1*b3*tck*v - A*E*a1*b3*tck - 2*A*E*a3*b1*tck*v)/d1
    K26 = (A*E*a1*a3*tck*v - A*E*a1*a3*tck - 2*A*E*b1*b3*tck)/d1

    K34 = (-A*E*a2*b2*tck)/d2
    K35 = (-2*A*E*a2*a3*tck + A*E*b2*b3*tck*v - A*E*b2*b3*tck)/d1
    K36 = (-2*A*E*a2*b3*tck*v + A*E*a3*b2*tck*v - A*E*a3*b2*tck)/d1

    K45 = (A*E*a2*b3*tck*v - A*E*a2*b3*tck - 2*A*E*a3*b2*tck*v)/d1
    K46 = (A*E*a2*a3*tck*v - A*E*a2*a3*tck - 2*A*E*b2*b3*tck)/d1

    K56 = (-A*E*a3*b3*tck)/d2

    
    K = array([
        [K11, K12, K13, K14, K15, K16],
        [K12, K22, K23, K24, K25, K26],
        [K13, K23, K33, K34, K35, K36],
        [K14, K24, K34, K44, K45, K46],
        [K15, K25, K35, K45, K55, K56],
        [K16, K26, K36, K46, K56, K66],
    ])
    
    return  K
    

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
    # Aeven = 1/2 * (x2*y3 - x3*y2 + y2*x1 - y3*x1 + x3*y1 - x2*y1) 
    A = 2 * (b*h)/(nelemx * nelemy)
    (aOdd, bOdd), (aEven, bEven) = getABC([x1, x2, x3], [y1, y2, y3], A), getABC([x1, x3, x4], [y1, y3, y4], A)
    Kodd = getK(aOdd, bOdd, A, E, v, tck)
    Keven = getK(aEven, bEven, A, E, v, tck)
    return (Kodd, Keven)

