import numpy as np
# import pandas as pd

# from get_neighborhood import get_neighborhood

# struct = pd.read_json("./python/triangularElements/str3x3.json")

def get_neighborhoodTrad(struct, rmin):
    """
    Vecindario para una malla triangular estructurada sobre un dominio rectangular.

    Parámetros
    ----------
    struct : dict u objeto
        Debe contener:
        - nodesx : número de nodos en x
        - nodesy : número de nodos en y
    rmin : float
        Radio mínimo de vecindad, en unidades de elemento.

    Retorna
    -------
    num_viz : np.ndarray, shape (n_elem,)
        Número de vecinos de cada elemento.
    N : np.ndarray, shape (max_num_viz, n_elem)
        Vecinos de cada elemento. La columna i contiene los vecinos de i.
        Las posiciones vacías se llenan con -1.
    D : np.ndarray, shape (max_num_viz, n_elem)
        Distancias de cada elemento a sus vecinos.
        Las posiciones vacías se llenan con np.nan.
    """

    def get_field(obj, name):
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    nodesx = int(get_field(struct, "nnodesx"))
    nodesy = int(get_field(struct, "nnodesy"))

    nelemx = nodesx - 1
    nelemy = nodesy - 1

    if nelemx <= 0 or nelemy <= 0:
        raise ValueError("nodesx y nodesy deben ser al menos 2.")

    n_elem = 2 * nelemx * nelemy

    def tri_centroid(i, j, tri):
        """
        Centroide del triángulo tri dentro de la celda rectangular (i, j),
        con i en [0, nelemx-1], j en [0, nelemy-1].

        tri = 0 -> triángulo inferior izquierdo
        tri = 1 -> triángulo superior derecho
        """
        if tri == 0:
            # Vértices: (i,j), (i+1,j), (i,j+1)
            return np.array([i - 1/3, j - 2/3], dtype=float)
        else:
            # Vértices: (i+1,j), (i+1,j+1), (i,j+1)
            return np.array([i - 2/3, j - 1/3], dtype=float)

    # Centroides de todos los triángulos
    centroids = np.zeros((n_elem, 2), dtype=float)

    e = 0
    for i in range(1, nelemx + 1 ):
        for j in range(1, nelemy + 1 ):
            centroids[e] = tri_centroid(i, j, 0)
            centroids[e + 1] = tri_centroid(i, j, 1)
            e += 2

    # Comparación tradicional: cada elemento contra todos los demás
    neighbors_per_elem = []
    distances_per_elem = []
    num_viz = np.zeros(n_elem, dtype=int)

    for i in range(n_elem):
        # print(f"{i + 1} Iteracion")
        diff = centroids - centroids[i]
        dist = np.linalg.norm(diff, axis=1)

        # print(dist)
        # print("")
        mask = (dist < rmin) 

        neigh = np.where(mask)[0]
        neigh = neigh + 1
        # print(neigh)

        # print("")
        # print(neigh)
        # print("")
        dvals = dist[mask]
        # # print(dvals)
        # # print("")

        # order = np.argsort(dvals)
        # print(order)
        # print("")
        # neigh = neigh[order]
        neighbors_per_elem.append(neigh)
        distances_per_elem.append(dvals)
        # print(neigh)
        # print("")
        # dvals = dvals[order]

        # neighbors_per_elem.append(neigh)
        # distances_per_elem.append(dvals)
        num_viz[i] = len(neigh)

    max_num_viz = int(num_viz.max()) if n_elem > 0 else 0
    # print(neighbors_per_elem[0])
    # N = -np.ones((max_num_viz, n_elem), dtype=int)
    N = np.zeros((max_num_viz, n_elem), dtype=int)
    D = np.zeros((max_num_viz, n_elem))
    for i in range(n_elem):
        end = len(neighbors_per_elem[i])
        N[0:end, i] = neighbors_per_elem[i]
        D[0:end, i] = distances_per_elem[i]
    # D = np.full((max_num_viz, n_elem), np.nan, dtype=float)

    # print(neighbors_per_elem)
    # print("")
    # for i in range(n_elem):
    #     k = num_viz[i]
    #     if k > 0:
    #         N[:k, i] = neighbors_per_elem[i]
    #         D[:k, i] = distances_per_elem[i]

    return num_viz, N, D


# num_viz, N, D = get_neighborhoodTrad(struct, 1.5)
# numNei, neighs, distnei = get_neighborhood(struct, 1.5)

# print(num_viz == )
# print(N)
# print( D )
# print("")
# print(distnei)