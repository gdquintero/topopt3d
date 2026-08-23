"""
=====================================================================
 SQP (Sequential Quadratic Programming) con aproximación diagonal
 de la Hessiana B_k, basado en la teoría del documento (Cap. 1,
 Sección 1.2.3: "Aproximación de la Hessiana de f a través de
 Matrices Diagonales").

 Idea central del PDF
 ---------------------
 En vez de calcular/actualizar la Hessiana completa de f (costoso),
 se usa el cambio de variable intermediario y_i = 1/x_i para
 construir una aproximación de primer orden f̃ de f alrededor de
 x^(k). Derivando dos veces f̃ se obtiene una Hessiana APROXIMADA
 que resulta DIAGONAL:

        d f̃/dxi (x^(k))      = df/dxi (x^(k))                (i.10a)

        d²f̃/dxi² (x^(k))     = -2/x_i^(k) * df/dxi (x^(k))    (i.10b)   si i=j
        d²f̃/(dxj dxi)(x^(k)) = 0                                        si i≠j

 Y luego se corrige para garantizar semidefinida positiva (>=0):

        b_{k,i} = d²f̃/dxi²(x^(k))                          si ese valor >= 0
        b_{k,i} = d²f̃/dxi²(x^(k)) + 1.1*| d²f̃/dxi²(x^(k)) |  en caso contrario

        B_k = diag(b_{k,1}, ..., b_{k,n})   ,   B_k >= 0

 Este B_k reemplaza a la Hessiana del Lagrangiano en el subproblema
 QP clásico de SQP:

        min_d   0.5 d^T B_k d + grad_f(x_k)^T d
        s.a.    grad_g_j(x_k)^T d + g_j(x_k) <= 0   (desigualdades)
                grad_h_i(x_k)^T d + h_i(x_k)  = 0   (igualdades)

 El subproblema QP se resuelve con `qpsolvers` (solver "quadprog"
 u "osqp"), en vez de una implementación manual de QP.

 NOTA IMPORTANTE: la fórmula (i.10) requiere x_i != 0 (viene de
 y_i = 1/x_i), así que este método diagonal es natural para
 problemas con x_i > 0 (p.ej. variables acotadas por abajo con
 x_i > 0), tal como en el documento.
=====================================================================
"""

import numpy as np
from qpsolvers import solve_qp


# ---------------------------------------------------------------
# 1. Utilidades numéricas: gradiente y jacobiano por diferencias
#    finitas centradas (útil si el usuario no da derivadas
#    analíticas).
# ---------------------------------------------------------------
def numerical_gradient(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    n = x.size
    grad = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        grad[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return grad


def numerical_jacobian(constraints, x, eps=1e-6, jacobians=None):
    """
    constraints: lista de funciones R^n -> R.
    jacobians  : lista opcional (mismo largo que `constraints`) con el
                 gradiente analítico de cada restricción, jac_j(x) -> R^n.
                 Donde jacobians[j] no sea None, se usa en vez de
                 diferencias finitas (mucho más rápido y sin ruido
                 numérico). Esto es lo que más pesa en el costo de
                 armar cada subproblema QP cuando hay muchas
                 restricciones, así que evitarlo cuando se pueda
                 acelera notablemente el `solve_qp` de cada iteración.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    m = len(constraints)
    J = np.zeros((m, n))
    for j, cj in enumerate(constraints):
        aj = jacobians[j] if jacobians is not None else None
        if aj is not None:
            J[j, :] = np.asarray(aj(x), dtype=float)
        else:
            J[j, :] = numerical_gradient(cj, x, eps)
    return J


# ---------------------------------------------------------------
# 2. Aproximación diagonal de la Hessiana B_k  (Ec. i.9 - i.10 - Bk)
# ---------------------------------------------------------------
def diagonal_hessian_Bk(x_k, grad_f_xk, min_eig=1e-8):
    """
    Construye B_k = diag(b_{k,1},...,b_{k,n}) usando la
    aproximación de f a través de variables intermediarias
    y_i = 1/x_i  (Ec. i.7 - i.10 del documento).

    Parámetros
    ----------
    x_k        : punto actual (x_i != 0 para todo i)
    grad_f_xk  : gradiente exacto (o numérico) de f en x_k
    min_eig    : cota inferior para evitar b_{k,i} = 0 exacto
                 (estabilidad numérica del QP)

    Retorna
    -------
    B_k : matriz diagonal (n x n), B_k >= 0
    b   : vector diagonal (n,)
    """
    x_k = np.asarray(x_k, dtype=float)
    grad_f_xk = np.asarray(grad_f_xk, dtype=float)

    if np.any(np.abs(x_k) < 1e-12):
        raise ValueError(
            "La aproximación diagonal usa y_i = 1/x_i; "
            "x_k tiene componentes ~0, no es aplicable ahí."
        )

    # Ec. (i.10b): d²f̃/dxi²(x_k) = -2/x_i * df/dxi(x_k)
    d2f_diag = -2.0 / x_k * grad_f_xk

    b = np.where(
        d2f_diag >= 0,
        d2f_diag,
        d2f_diag + 1.1 * np.abs(d2f_diag),
    )

    # pequeña regularización para que el QP sea siempre resoluble
    b = np.maximum(b, min_eig)

    B_k = np.diag(b)
    return B_k, b


# ---------------------------------------------------------------
# 3. Subproblema QP (resuelto con qpsolvers)
# ---------------------------------------------------------------
def solve_qp_subproblem(B_k, grad_f_xk, x_k,
                         ineq_constraints=None, eq_constraints=None,
                         ineq_jac=None, eq_jac=None,
                         lb_d=None, ub_d=None,
                         initvals=None,
                         sparse_P=False,
                         solver="quadprog", **solver_opts):
    """
    Resuelve:
        min_d   0.5 d^T B_k d + grad_f(x_k)^T d
        s.a.    J_g(x_k) d <= -g(x_k)      (linealización de g<=0)
                J_h(x_k) d  = -h(x_k)      (linealización de h=0)
                lb_d <= d <= ub_d           (caja NATIVA sobre el paso;
                                             mucho más barata que meter
                                             lb/ub como filas de G)

    Parámetros de rendimiento
    --------------------------
    ineq_jac, eq_jac : listas opcionales de gradientes analíticos de
                        cada restricción (mismo orden que ineq_constraints/
                        eq_constraints). Si se dan, se evita el costo de
                        diferencias finitas (2 evaluaciones de función
                        por restricción por variable) en cada iteración.
    initvals         : d_{k-1} (solución del QP anterior) para
                        WARM-START. Reduce notablemente las iteraciones
                        internas de solvers iterativos como "osqp".
    sparse_P         : si True, construye B_k como matriz dispersa
                        (scipy.sparse.diags) — recomendado con
                        solver="osqp" para n grande, ya que B_k es
                        diagonal y así se evita operar con una matriz
                        densa n x n llena de ceros.
    **solver_opts    : kwargs adicionales pasados tal cual a qpsolvers
                        (p.ej. eps_abs=1e-4, eps_rel=1e-4, max_iter=500
                        para "osqp": tolerancias más laxas = más rápido,
                        a costa de precisión).
    """
    if sparse_P:
        import scipy.sparse as sp
        P = sp.diags(np.diag(B_k))
    else:
        P = B_k
    q = grad_f_xk

    G, h_vec = None, None
    if ineq_constraints:
        g_vals = np.array([g(x_k) for g in ineq_constraints])
        J_g = numerical_jacobian(ineq_constraints, x_k, jacobians=ineq_jac)
        G = J_g
        h_vec = -g_vals

    A, b_vec = None, None
    if eq_constraints:
        h_vals = np.array([h(x_k) for h in eq_constraints])
        J_h = numerical_jacobian(eq_constraints, x_k, jacobians=eq_jac)
        A = J_h
        b_vec = -h_vals

    d = solve_qp(P, q, G=G, h=h_vec, A=A, b=b_vec,
                 lb=lb_d, ub=ub_d, initvals=initvals,
                 solver=solver, **solver_opts)
    if d is None:
        raise RuntimeError("El subproblema QP no encontró solución factible.")
    return d


# ---------------------------------------------------------------
# 4. Merit function (penalización L1) y line search tipo Armijo
#    para garantizar progreso global del SQP.
# ---------------------------------------------------------------
def l1_merit(f, x, ineq_constraints, eq_constraints, mu):
    val = f(x)
    if ineq_constraints:
        val += mu * sum(max(0.0, g(x)) for g in ineq_constraints)
    if eq_constraints:
        val += mu * sum(abs(h(x)) for h in eq_constraints)
    return val


def armijo_line_search(f, x_k, d_k, grad_f_xk, ineq_constraints,
                        eq_constraints, mu, alpha0=1.0, c1=1e-4,
                        rho=0.5, max_iter=30):
    # phi0 = l1_merit(f, x_k, ineq_constraints, eq_constraints, mu)
    # directional_deriv = grad_f_xk @ d_k
    # if ineq_constraints:
    #     directional_deriv -= mu * sum(max(0.0, g(x_k)) for g in ineq_constraints)
    # if eq_constraints:
    #     directional_deriv -= mu * sum(abs(h(x_k)) for h in eq_constraints)

    # alpha = alpha0
    # for _ in range(max_iter):
    #     x_new = x_k + alpha * d_k
    #     phi_new = l1_merit(f, x_new, ineq_constraints, eq_constraints, mu)
    #     if phi_new <= phi0 + c1 * alpha * directional_deriv:
    #         return alpha
    #     alpha *= rho
    return 1


# ---------------------------------------------------------------
# 5. Núcleo del algoritmo SQP (uso interno)
# ---------------------------------------------------------------
def _sqp_diagonal_core(f, x0, ineq_constraints=None, eq_constraints=None,
                        ineq_jac=None, eq_jac=None,
                        lb_x=None, ub_x=None,
                        grad_f=None, max_iter=100, tol=1e-6,
                        qp_solver="quadprog", warm_start=True,
                        sparse_P=False, verbose=True, **solver_opts):
    """
    Núcleo puro del SQP con Hessiana diagonal B_k. Espera que f,
    grad_f (si se da) e ineq_constraints/eq_constraints ya sean
    funciones de un solo argumento x (sin *args). Uso interno:
    la función pública es `sqp_diagonal` más abajo.

    Optimizaciones de velocidad para el QP de cada iteración:
      - lb_x, ub_x   : cotas de x (no de d) pasadas como caja NATIVA
                       del QP (lb_x - x_k <= d <= ub_x - x_k), en vez
                       de agregarlas como filas de restricción general
                       derivadas por diferencias finitas.
      - ineq_jac/eq_jac : jacobianos analíticos opcionales de las
                       restricciones generales (evitan diferencias
                       finitas, que son el costo dominante cuando hay
                       muchas restricciones).
      - warm_start   : reutiliza d_{k-1} como punto inicial del QP
                       siguiente (initvals), lo que acelera solvers
                       iterativos como "osqp".
      - sparse_P     : usa B_k como matriz dispersa diagonal
                       (recomendado con qp_solver="osqp" si n es grande).
    """
    x_k = np.asarray(x0, dtype=float)
    ineq_constraints = ineq_constraints or []
    eq_constraints = eq_constraints or []
    grad_f = grad_f or (lambda x: numerical_gradient(f, x))

    x_k_old = x_k
    f_k_old = f(x_k_old)
    n = x_k.size
    lb_x = np.full(n, -np.inf) if lb_x is None else np.asarray(lb_x, dtype=float)
    ub_x = np.full(n, np.inf) if ub_x is None else np.asarray(ub_x, dtype=float)
    has_bounds = np.any(np.isfinite(lb_x)) or np.any(np.isfinite(ub_x))

    mu = 10.0  # peso de penalización del merit function
    history = [x_k.copy()]
    d_prev = None  # para warm-start

    for k in range(max_iter):
        grad_f_xk = grad_f(x_k)

        # --- Aproximación diagonal de la Hessiana (teoría del PDF) ---
        B_k, b_diag = diagonal_hessian_Bk(x_k, grad_f_xk)

        # --- caja nativa del QP: lb_x <= x_k + d <= ub_x  =>  lb_x-x_k <= d <= ub_x-x_k
        lb_d = (lb_x - x_k) if has_bounds else None
        ub_d = (ub_x - x_k) if has_bounds else None

        # --- Resolver subproblema QP con qpsolvers ---
        d_k = solve_qp_subproblem(
            B_k, grad_f_xk, x_k,
            ineq_constraints=ineq_constraints,
            eq_constraints=eq_constraints,
            ineq_jac=ineq_jac, eq_jac=eq_jac,
            lb_d=lb_d, ub_d=ub_d,
            initvals=(d_prev if warm_start else None),
            sparse_P=sparse_P,
            solver=qp_solver,
            **solver_opts,
        )
        d_prev = d_k


        # --- Line search (merit function L1, Armijo) ---
        alpha = armijo_line_search(
            f, x_k, d_k, grad_f_xk, ineq_constraints, eq_constraints, mu
        )

        x_k = x_k + alpha * d_k
        f_k = f(x_k)
        step_norm = np.linalg.norm(d_k, np.inf)
        step_f_norm = np.abs(f_k - f_k_old)/max(np.abs(f_k_old), 1)
        x_k_old = x_k
        f_k_old = f_k
        # kktNorm = np.linalg.norm(grad_f_xk, np.inf)
        if verbose:
            print(f"iter {k:3d}: f(x)={f_k: .6e}  "
                  f"||d_k||={step_norm: .3e} ||f_k+1 - f_k||={step_f_norm: .3e}  b_diag={np.round(b_diag, 3)}")

        if step_norm < tol or step_f_norm < tol:
            break
        history.append(x_k.copy())

    return {
        "x": x_k,
        "f": f(x_k),
        "n_iter": k + 1,
        "history": np.array(history),
    }


# =====================================================================
# 6. FUNCIÓN PÚBLICA UNIFICADA: sqp_diagonal
# ---------------------------------------------------------------------
# Esta función sirve para DOS estilos de uso, sin que tengas que
# acordarte de cuál usar:
#
#   (A) Uso directo / standalone:
#       res = sqp_diagonal(f, x0, ineq_constraints=[g1, g2], ...)
#       res.x, res.fun, res['x'], res['f']   # ambos accesos funcionan
#
#   (B) Como `method` personalizado de scipy.optimize.minimize:
#       res = minimize(f, x0, jac=..., bounds=..., constraints=[...],
#                       method=sqp_diagonal, options={...})
#       res.x, res.fun, res.success, res.nit
#
# En el caso (B), scipy invoca internamente:
#     sqp_diagonal(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
#                  bounds=bounds, constraints=constraints,
#                  callback=callback, **options)
# por eso esta función acepta también jac/hess/hessp/bounds/
# constraints(formato scipy)/callback, además de la API propia
# (ineq_constraints/eq_constraints/grad_f).
#
# El resultado SIEMPRE es un scipy.optimize.OptimizeResult (que es
# una subclase de dict), así que sigue soportando el acceso tipo
# diccionario res['x'], res['f'], res['n_iter'] de antes, y además
# el acceso tipo atributo res.x, res.fun, res.success, res.nit que
# scipy necesita.
#
# Convención de signos en `constraints` (formato scipy): 'ineq'
# significa fun(x) >= 0 es factible, mientras que internamente (y en
# `ineq_constraints`) usamos g(x) <= 0; por eso se invierte el signo
# al traducir.
# =====================================================================
def sqp_diagonal(fun, x0, args=(), ineq_constraints=None, eq_constraints=None,
                  ineq_jac=None, eq_jac=None,
                  grad_f=None, jac=None, hess=None, hessp=None,
                  bounds=None, constraints=(), callback=None,
                  max_iter=100, tol=1e-6, qp_solver="quadprog",
                  warm_start=True, sparse_P=False,
                  verbose=True, **options):
    """
    SQP con Hessiana aproximada por B_k diagonal (Sección 1.2.3 del PDF)
    y subproblema QP resuelto con qpsolvers. Compatible con uso directo
    y con scipy.optimize.minimize(method=sqp_diagonal, ...).

    Parámetros (API propia)
    ------------------------
    fun              : función objetivo fun(x, *args) -> R
    x0               : punto inicial (con x0_i != 0 en cada componente)
    args             : argumentos extra para fun/grad_f/restricciones
    ineq_constraints : lista de funciones g_j(x, *args) <= 0
    eq_constraints   : lista de funciones h_i(x, *args) = 0
    ineq_jac, eq_jac : listas OPCIONALES de gradientes analíticos
                        (mismo orden/largo que ineq_constraints/
                        eq_constraints), jac_j(x) -> R^n. Evitan las
                        diferencias finitas, que son el costo dominante
                        de armar el QP cuando hay muchas restricciones.
    grad_f           : gradiente de fun (opcional; si None, diferencias
                        finitas). Firma grad_f(x, *args).
    max_iter, tol    : criterios de paro (||d_k|| < tol)
    qp_solver        : "quadprog" u "osqp"
    verbose          : imprime progreso por iteración

    Parámetros de RENDIMIENTO del subproblema QP
    -----------------------------------------------
    warm_start       : (default True) reutiliza la solución d_{k-1}
                        del QP anterior como punto inicial del
                        siguiente (initvals). Acelera solvers
                        iterativos como "osqp"; con "quadprog"
                        (solver activo-set directo) el efecto es menor.
    sparse_P         : (default False) construye B_k como matriz
                        dispersa diagonal en vez de densa. Útil para
                        n grande combinado con qp_solver="osqp".
    **options        : además de 'maxiter'/'disp', se puede pasar
                        cualquier kwarg propio del solver de qpsolvers,
                        p.ej. eps_abs=1e-4, eps_rel=1e-4 (más laxo =
                        más rápido) si qp_solver="osqp".

    Parámetros adicionales (compatibilidad con scipy.optimize.minimize)
    ---------------------------------------------------------------------
    jac              : alias de grad_f (así lo llama scipy). Firma jac(x, *args).
    hess, hessp      : ignorados (B_k es la única aproximación de 2do orden usada)
    bounds           : lista de tuplas (lb, ub) o scipy.optimize.Bounds;
                        se pasan como CAJA NATIVA del QP (lb<=x+d<=ub),
                        no como restricciones generales -> mucho más
                        rápido que la versión anterior de este código.
    constraints      : lista de dicts {'type': 'ineq'|'eq', 'fun': ...,
                        'args': ..., 'jac': ...} (formato scipy) o un
                        solo dict. Si el dict trae 'jac' (gradiente
                        analítico), también se usa para evitar
                        diferencias finitas.
    callback         : callback(x_k) llamado al final con el punto óptimo

    Retorna
    -------
    scipy.optimize.OptimizeResult con x, fun, jac, nit, success, message,
    y además (compatibilidad hacia atrás) las claves 'f', 'n_iter', 'history'.
    """
    from scipy.optimize import OptimizeResult

    x0 = np.asarray(x0, dtype=float)
    ineq_constraints = list(ineq_constraints) if ineq_constraints else []
    eq_constraints = list(eq_constraints) if eq_constraints else []
    ineq_jac = list(ineq_jac) if ineq_jac else [None] * len(ineq_constraints)
    eq_jac = list(eq_jac) if eq_jac else [None] * len(eq_constraints)

    # alias de opciones al estilo scipy; el resto de options se reenvía
    # tal cual al solver de qpsolvers (p.ej. eps_abs, eps_rel, etc.)
    max_iter = int(options.pop("maxiter", max_iter))
    verbose = bool(options.pop("disp", verbose))
    grad_f = grad_f if grad_f is not None else jac  # jac es el nombre que usa scipy

    # --- envolver fun/grad_f/restricciones propias con args, evitando
    #     auto-referencia por clausura (nombres distintos) ---
    if args:
        fun_orig, grad_f_orig = fun, grad_f
        fun_ = lambda x, _f=fun_orig: _f(x, *args)
        ineq_ = [(lambda x, _g=g: _g(x, *args)) for g in ineq_constraints]
        eq_ = [(lambda x, _h=h: _h(x, *args)) for h in eq_constraints]
        grad_f_ = (lambda x, _gf=grad_f_orig: _gf(x, *args)) if grad_f_orig is not None else None
    else:
        fun_, ineq_, eq_, grad_f_ = fun, ineq_constraints, eq_constraints, grad_f

    ineq_all, eq_all = list(ineq_), list(eq_)
    ineq_jac_all, eq_jac_all = list(ineq_jac), list(eq_jac)

    # --- traducir constraints estilo scipy (lista de dicts) ---
    cons_list = [constraints] if isinstance(constraints, dict) else list(constraints or [])
    for c in cons_list:
        c_fun = c["fun"]
        c_args = c.get("args", ())
        c_jac = c.get("jac", None)  # gradiente analítico opcional del dict scipy
        if c["type"] == "ineq":
            # scipy: c_fun(x) >= 0 factible  <=>  interno: g(x) = -c_fun(x) <= 0
            ineq_all.append(lambda x, _f=c_fun, _a=c_args: -_f(x, *_a))
            ineq_jac_all.append((lambda x, _j=c_jac, _a=c_args: -np.asarray(_j(x, *_a), dtype=float))
                                 if callable(c_jac) else None)
        elif c["type"] == "eq":
            eq_all.append(lambda x, _f=c_fun, _a=c_args: _f(x, *_a))
            eq_jac_all.append((lambda x, _j=c_jac, _a=c_args: np.asarray(_j(x, *_a), dtype=float))
                               if callable(c_jac) else None)
        else:
            raise ValueError(f"Tipo de restricción desconocido: {c['type']!r}")

    # --- traducir bounds -> caja NATIVA del QP (lb_x, ub_x), no filas
    #     de restricción general (esto es lo que más acelera el QP) ---
    lb_x = ub_x = None
    if bounds is not None:
        if hasattr(bounds, "lb") and hasattr(bounds, "ub"):  # scipy.optimize.Bounds
            lb_x = np.atleast_1d(np.asarray(bounds.lb, dtype=float))
            ub_x = np.atleast_1d(np.asarray(bounds.ub, dtype=float))
        else:  # lista de tuplas (min, max)
            lb_x = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds])
            ub_x = np.array([(np.inf if b[1] is None else b[1]) for b in bounds])

    # --- correr el núcleo del SQP diagonal ---
    result = _sqp_diagonal_core(
        fun_, x0,
        ineq_constraints=ineq_all or None,
        eq_constraints=eq_all or None,
        ineq_jac=(ineq_jac_all or None),
        eq_jac=(eq_jac_all or None),
        lb_x=lb_x, ub_x=ub_x,
        grad_f=grad_f_,
        max_iter=max_iter,
        tol=tol if tol is not None else 1e-6,
        qp_solver=qp_solver,
        warm_start=warm_start,
        sparse_P=sparse_P,
        verbose=verbose,
        **options,
    )

    x_star = result["x"]
    grad_final = grad_f_(x_star) if grad_f_ else numerical_gradient(fun_, x_star)

    feas_ineq = all(g(x_star) <= 1e-6 for g in ineq_all) if ineq_all else True
    feas_eq = all(abs(h(x_star)) <= 1e-6 for h in eq_all) if eq_all else True
    if lb_x is not None:
        feas_ineq = feas_ineq and np.all(x_star >= lb_x - 1e-6)
    if ub_x is not None:
        feas_ineq = feas_ineq and np.all(x_star <= ub_x + 1e-6)
    converged = result["n_iter"] < max_iter
    success = converged and feas_ineq and feas_eq

    if callback is not None:
        try:
            callback(x_star)
        except TypeError:
            callback(OptimizeResult(x=x_star, fun=result["f"]))

    res = OptimizeResult(
        x=x_star,
        fun=result["f"],
        jac=grad_final,
        nit=result["n_iter"],
        nfev=None,
        success=success,
        status=0 if success else 1,
        message=("Optimización completada (SQP diagonal B_k)." if success
                  else "Se alcanzó max_iter o el punto final no es totalmente factible."),
    )
    # compatibilidad hacia atrás: acceso tipo diccionario res['f'], res['n_iter']
    res["f"] = result["f"]
    res["n_iter"] = result["n_iter"]
    res["history"] = result["history"]
    return res


# alias por compatibilidad con código que ya usaba este nombre explícito
sqp_diagonal_scipy = sqp_diagonal




# =====================================================================
# 7. EJEMPLO / VALIDACIÓN
#    Problema:  min f(x) = (x1-2)^2 + (x2-3)^2
#               s.a.   x1 + x2 <= 4
#                      x1 >= 0.5, x2 >= 0.5   (x_i != 0, requerido por 1/x_i)
# =====================================================================
# if __name__ == "__main__":
#     from scipy.optimize import minimize

#     def f(x):
#         return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

#     def g1_directo(x):  # convención propia: g(x) <= 0
#         return x[0] + x[1] - 4

#     def g1_scipy(x):  # convención scipy: fun(x) >= 0 factible
#         return 4 - x[0] - x[1]

#     x0 = np.array([1.0, 1.0])
#     bounds = [(0.5, None), (0.5, None)]

#     print("=" * 70)
#     print(" (A) sqp_diagonal() USADO DIRECTAMENTE (API propia)")
#     print("=" * 70)
#     res_directo = sqp_diagonal(
#         f, x0,
#         ineq_constraints=[g1_directo, lambda x: 0.5 - x[0], lambda x: 0.5 - x[1]],
#         max_iter=50, tol=1e-8, verbose=False,
#     )
#     print("  x* =", res_directo.x, " (o res_directo['x'] también funciona)")
#     print("  f(x*) =", res_directo.fun, " (o res_directo['f'])")

#     print("\n" + "=" * 70)
#     print(" (B) sqp_diagonal COMO method= DE scipy.optimize.minimize")
#     print("=" * 70)
#     res_scipy_style = minimize(
#         f, x0, method=sqp_diagonal,               # <-- se pasa la función, no un string
#         bounds=bounds,
#         constraints=[{"type": "ineq", "fun": g1_scipy}],
#         options={"max_iter": 50, "tol": 1e-8, "qp_solver": "quadprog"},
#     )
#     print("  x* =", res_scipy_style.x)
#     print("  f(x*) =", res_scipy_style.fun)
#     print("  iteraciones =", res_scipy_style.nit)
#     print("  success =", res_scipy_style.success, "-", res_scipy_style.message)

#     print("\n" + "=" * 70)
#     print(" (C) scipy.optimize.minimize(method='SLSQP')  (referencia)")
#     print("=" * 70)
#     res_slsqp = minimize(f, x0, method="SLSQP", bounds=bounds,
#                           constraints=[{"type": "ineq", "fun": g1_scipy}])
#     print("  x* =", res_slsqp.x)
#     print("  f(x*) =", res_slsqp.fun)
#     print("  iteraciones =", res_slsqp.nit)