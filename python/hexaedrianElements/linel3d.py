import numpy as np


def linel3d(struct):
    nelx, nely = int(struct.nelx), int(struct.nely)
    lin = np.zeros( (24 ,int(struct.nelem)))
    floor = 0
    for i in range(1, int(struct.nelem) + 1):
        node = i + int(i/nelx) + floor*struct.nodesx if i% (nelx)  != 0 else i + int(i/nelx) + floor*struct.nodesx  - 1
        if (nelx * nely*(floor+1)) == i:
            floor += 1


        lin[2 , i - 1] = 3*node 
        lin[1, i - 1] = lin[2 , i - 1] - 1
        lin[0, i - 1] = lin[1, i - 1] - 1

        lin[5, i - 1] = lin[2, i-1] + 3*struct.nodesx
        lin[4, i - 1] = lin[5, i-1] -1
        lin[3, i - 1] = lin[4, i-1] -1

        lin[8, i-1] = lin[5, i-1] + 3
        lin[7, i-1] = lin[8, i-1] - 1
        lin[6, i-1] = lin[7, i-1] - 1
        
        lin[11, i-1] = lin[2, i-1] + 3
        lin[10, i-1] = lin[11, i-1] - 1
        lin[9, i-1] = lin[10, i-1] - 1

        lin[14, i-1] = lin[2, i-1] + 3*struct.nodesx*struct.nodesy
        lin[13, i-1] = lin[14, i-1] - 1
        lin[12, i-1] = lin[13, i-1] - 1

        lin[17, i-1] = lin[2, i-1] + 3*struct.nodesx*struct.nodesy + 3*struct.nodesx
        lin[16, i-1] = lin[17, i-1] - 1
        lin[15, i-1] = lin[16, i-1] - 1
        
        lin[20, i-1] = lin[17, i-1] + 3
        lin[19, i-1] = lin[20, i-1] - 1
        lin[18, i-1] = lin[19, i-1] - 1

        lin[23, i-1] = lin[14, i-1] + 3
        lin[22, i-1] = lin[23, i-1] - 1
        lin[21, i-1] = lin[22, i-1] - 1        

    return lin
        

