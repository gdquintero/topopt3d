from numpy import zeros

def put_supports(struct):
    linap = zeros(2*len(struct.supp[0]["node"]))
    counter = 0

    for i in range(len(struct.supp[0]["node"])):
        if struct.supp[0]["ix"][i] == 1:
            linap[counter] = 2*struct.supp[0]["node"][i] - 1
            counter += 1
        if struct.supp[0]["iy"][i] == 1:
            linap[counter] = 2*struct.supp[0]["node"][i] 
            counter += 1
    return linap

