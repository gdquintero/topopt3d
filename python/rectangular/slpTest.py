"""
custom_slp.py
=============

Programación Lineal Secuencial (SLP) con REGIÓN DE CONFIANZA, implementada
"a mano", con interfaz compatible con scipy.optimize.minimize (misma familia
que custom_sqp.py y custom_mma.py).

Arquitectura
------------
1. En cada iteración se linealiza TODO alrededor de x^(k):
        f(x+d)      ≈ f(x) + grad_f(x)^T d
        c_eq(x+d)   ≈ c_eq(x) + A_eq d        (= 0)
        c_ineq(x+d) ≈ c_ineq(x) + A_ineq d    (>= 0)

   y se resuelve el subproblema LINEAL (LP):
        min_d   grad_f(x)^T d
        s.t.    A_eq d = -c_eq(x)
                A_ineq d >= -c_ineq(x)
                d in [max(xmin-x, -Δ), min(xmax-x, Δ)]   (región de confianza)

   Como el objetivo linealizado no tiene curvatura, es imprescindible acotar
   d con una región de confianza Δ (si no, la LP sería ilimitada o daría
   pasos absurdamente grandes). La LP en sí se resuelve con
   scipy.optimize.linprog (un primitivo de álgebra lineal, no un solver de
   NLP) -- toda la lógica de SLP (construir el modelo linealizado, controlar
   Δ, aceptar/rechazar pasos) es propia.

2. GLOBALIZACIÓN por región de confianza clásica: se compara la reducción
   real de una función de mérito ℓ1 contra la reducción PREDICHA por el
   modelo lineal.
        ared = phi(x) - phi(x+d)
        pred = phi(x) - [modelo lineal de phi en x+d]
        rho  = ared / pred
   Si rho es bueno (>= eta2) se acepta el paso y se agranda Δ; si es
   aceptable (>= eta1) se acepta y Δ se mantiene; si es malo, se rechaza el
   paso y se contrae Δ (sin mover x). Esto es lo que hace que SLP converja
   de forma confiable pese a no tener curvatura de segundo orden.

3. Convergencia: paso pequeño + factibilidad, similar a custom_sqp.py.

Convención de restricciones (igual que scipy / custom_sqp / custom_mma):
    constraints = [
        {'type': 'eq',   'fun': c, 'jac': dc},
        {'type': 'ineq', 'fun': c, 'jac': dc},   # convención: c(x) >= 0
    ]

bounds es recomendable (aunque no estrictamente obligatorio como en MMA): si
faltan cotas en alguna variable, se usa una región de confianza inicial fija
(delta0) para esa variable, ya que SLP la necesita de todas formas.
"""

import numpy as np
from scipy.optimize import linprog
import warnings


class SLPResult:
    """Contenedor de resultados, similar a scipy.optimize.OptimizeResult."""
    def __init__(self, x, fun, success, message, nit, history):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nit = nit
        self.history = history

    def __repr__(self):
        return (f"SLPResult(success={self.success}, fun={self.fun:.8g}, "
                f"nit={self.nit}, message='{self.message}')")


# ----------------------------------------------------------------------
# Utilidades para apilar restricciones (igual estilo que custom_sqp.py)
# ----------------------------------------------------------------------
def _stack_constraints(constraints, ctype, x):
    funs = [c for c in constraints if c['type'] == ctype]
    if not funs:
        return None, None
    vals, jacs = [], []
    for c in funs:
        v = np.atleast_1d(np.asarray(c['fun'](x), dtype=float))
        J = np.atleast_2d(np.asarray(c['jac'](x), dtype=float))
        if J.shape[0] != v.shape[0]:
            J = J.reshape(v.shape[0], -1)
        vals.append(v)
        jacs.append(J)
    return np.concatenate(vals), np.vstack(jacs)


def _merit(fun, constraints, x, sigma):
    f = fun(x)
    c_eq, _ = _stack_constraints(constraints, 'eq', x)
    c_ineq, _ = _stack_constraints(constraints, 'ineq', x)
    viol = 0.0
    if c_eq is not None:
        viol += np.sum(np.abs(c_eq))
    if c_ineq is not None:
        viol += np.sum(np.maximum(0.0, -c_ineq))
    return f + sigma * viol, viol, c_eq, c_ineq


def _predicted_merit(f, g, d, c_eq, A_eq, c_ineq, A_ineq, sigma):
    """Valor del modelo LINEAL de la función de mérito en x+d."""
    f_lin = f + g @ d
    viol_lin = 0.0
    if c_eq is not None:
        viol_lin += np.sum(np.abs(c_eq + A_eq @ d))
    if c_ineq is not None:
        viol_lin += np.sum(np.maximum(0.0, -(c_ineq + A_ineq @ d)))
    return f_lin + sigma * viol_lin


# ----------------------------------------------------------------------
# Subproblema LP (linprog)
# ----------------------------------------------------------------------
def _solve_lp(g, A_eq, b_eq, A_ineq, b_ineq, d_lb, d_ub):
    """
    min g^T d  s.t. A_eq d = b_eq, A_ineq d >= b_ineq, d_lb <= d <= d_ub
    (resuelto con scipy.optimize.linprog, method='highs')
    Devuelve d, y_eq (duales de igualdad), y_ineq (duales de desigualdad).
    """
    n = g.size
    A_ub = -A_ineq if A_ineq is not None else None
    b_ub = -b_ineq if b_ineq is not None else None
    bounds = list(zip(d_lb, d_ub))

    res = linprog(g, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method='highs')
    if not res.success:
        # LP infactible dentro de la región de confianza: devolver paso nulo
        # (el control de Δ en el lazo externo se encargará de ajustar).
        return np.zeros(n), np.zeros(0 if b_eq is None else len(b_eq)), \
               np.zeros(0 if b_ineq is None else len(b_ineq)), False

    d = res.x
    y_eq = -res.eqlin.marginals if (b_eq is not None and hasattr(res, 'eqlin')) else np.zeros(0)
    y_ineq = res.ineqlin.marginals if (b_ub is not None and hasattr(res, 'ineqlin')) else np.zeros(0)
    # marginals de linprog vienen con signo de A_ub d <= b_ub; los reescalamos
    # a la convención A_ineq d >= b_ineq (mu >= 0 en el óptimo)
    y_ineq = -y_ineq if y_ineq.size else y_ineq
    return d, y_eq, y_ineq, True


# ----------------------------------------------------------------------
# Solver principal
# ----------------------------------------------------------------------
def minimize_slp(fun, x0, jac, constraints=None, bounds=None,
                  maxiter=200, tol=1e-6, delta0=None,
                  delta_min=1e-8, delta_max=None,
                  eta1=0.25, eta2=0.75, shrink=0.25, expand=2.0,
                  verbose=False):
    """
    Minimiza fun(x) sujeto a restricciones de igualdad/desigualdad y bounds,
    usando Programación Lineal Secuencial (SLP) con región de confianza.

    Parameters
    ----------
    fun : callable(x) -> float
    x0  : array_like
    jac : callable(x) -> array   (gradiente analítico, obligatorio)
    constraints : list of dict {'type': 'eq'|'ineq', 'fun':.., 'jac':..}
        Convención 'ineq': fun(x) >= 0.
    bounds : list of (lb, ub) o None.
    maxiter, tol : criterios de paro (norma del paso y factibilidad).
    delta0 : radio inicial de la región de confianza por variable
        (por defecto: 10% del rango [lb,ub], o 1.0 si no hay bounds).
    delta_min, delta_max : límites del radio de confianza.
    eta1, eta2 : umbrales de la razón de reducción real/predicha para
        aceptar el paso (eta1) y para agrandar Δ (eta2).
    shrink, expand : factores de contracción/expansión de Δ.
    verbose : bool

    Returns
    -------
    SLPResult
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    constraints = constraints or []

    if bounds is not None:
        xmin = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
        xmax = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
    else:
        xmin = np.full(n, -np.inf)
        xmax = np.full(n, np.inf)

    if delta0 is None:
        rng = xmax - xmin
        delta = np.where(np.isfinite(rng), 0.1 * rng, 1.0)
    else:
        delta = np.full(n, float(delta0)) if np.isscalar(delta0) else np.asarray(delta0, dtype=float).copy()

    if delta_max is None:
        rng = xmax - xmin
        delta_max_arr = np.where(np.isfinite(rng), rng, 1e3)
    else:
        delta_max_arr = np.full(n, float(delta_max)) if np.isscalar(delta_max) else np.asarray(delta_max, dtype=float)

    sigma = 1.0
    history = []
    success = False
    message = "Máximo de iteraciones alcanzado"
    it = 0

    while it < maxiter:
        it += 1
        f = fun(x)
        g = np.asarray(jac(x), dtype=float)
        c_eq, A_eq = _stack_constraints(constraints, 'eq', x)
        c_ineq, A_ineq = _stack_constraints(constraints, 'ineq', x)

        b_eq = -c_eq if c_eq is not None else None
        b_ineq = -c_ineq if c_ineq is not None else None

        d_lb = np.maximum(xmin - x, -delta)
        d_ub = np.minimum(xmax - x, delta)
        d_lb = np.minimum(d_lb, 0.0)   # asegurar 0 sea factible en la caja
        d_ub = np.maximum(d_ub, 0.0)

        d, y_eq, y_ineq, lp_ok = _solve_lp(g, A_eq, b_eq, A_ineq, b_ineq, d_lb, d_ub)

        step_norm = np.max(np.abs(d)) if d.size else 0.0
        feas_eq = 0.0 if c_eq is None else np.max(np.abs(c_eq))
        feas_ineq = 0.0 if c_ineq is None else max(0.0, -np.min(c_ineq))

        history.append(dict(it=it, f=f, x=x.copy(), step_norm=step_norm,
                             feas_eq=feas_eq, feas_ineq=feas_ineq, delta=delta.copy()))

        if verbose:
            print(f"it={it:3d}  f={f:.8g}  |d|_inf={step_norm:.3e}  "
                  f"feas_eq={feas_eq:.3e}  feas_ineq={feas_ineq:.3e}  "
                  f"delta_avg={np.mean(delta):.3e}")

        if (lp_ok and step_norm < tol and feas_eq < tol and feas_ineq < tol):
            success = True
            message = "Convergencia (paso despreciable y restricciones satisfechas)"
            break

        if not lp_ok:
            # región de confianza demasiado restrictiva / infactible: agrandar y reintentar
            delta = np.minimum(delta * expand, delta_max_arr)
            continue

        # --- actualizar sigma (peso de penalización de la función de mérito) ---
        max_mult = 0.0
        if y_eq.size:
            max_mult = max(max_mult, np.max(np.abs(y_eq)))
        if y_ineq.size:
            max_mult = max(max_mult, np.max(np.abs(y_ineq)))
        sigma = max(sigma, max_mult + 1.0)

        # --- razón de reducción real / predicha (control de región de confianza) ---
        phi0, viol0, _, _ = _merit(fun, constraints, x, sigma)
        pred0 = phi0  # el modelo lineal en d=0 coincide con phi0
        pred_new = _predicted_merit(f, g, d, c_eq, A_eq, c_ineq, A_ineq, sigma)
        pred_reduction = pred0 - pred_new

        phi_new, viol_new, _, _ = _merit(fun, constraints, x + d, sigma)
        actual_reduction = phi0 - phi_new

        if pred_reduction <= 1e-14:
            rho = -1.0  # sin mejora predicha: tratar como paso malo
        else:
            rho = actual_reduction / pred_reduction

        if rho < eta1:
            # paso rechazado: no mover x, contraer la región de confianza
            delta = np.maximum(delta * shrink, delta_min)
            continue

        # paso aceptado
        x = x + d
        if rho >= eta2:
            delta = np.minimum(delta * expand, delta_max_arr)
        # si eta1 <= rho < eta2: mantener delta

    return SLPResult(x=x, fun=fun(x), success=success, message=message,
                      nit=it, history=history)


# ----------------------------------------------------------------------
# Adaptador para usarlo como method= en scipy.optimize.minimize
# ----------------------------------------------------------------------
def slp_method(fun, x0, args=(), jac=None, bounds=None, constraints=(),
               callback=None, maxiter=200, tol=1e-6, delta0=0.01,
               delta_min=1e-8, delta_max=None, eta1=0.25, eta2=0.75,
               shrink=0.25, expand=2.0, verbose=False, **unused_options):
    """
    Adaptador SLP compatible con la interfaz de "custom minimizer" de scipy:

        from scipy.optimize import minimize
        from custom_slp import slp_method

        res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
                        method=slp_method, options={'maxiter': 200, 'tol': 1e-6})

    Soporta args y callback igual que sqp_method / mma_method.
    """
    from scipy.optimize import OptimizeResult

    if jac is None:
        raise ValueError("slp_method requiere 'jac' (gradiente analítico).")

    def fun_(x):
        return fun(x, *args)

    def jac_(x):
        return np.asarray(jac(x, *args), dtype=float)

    constraints = constraints if constraints else []
    if isinstance(constraints, dict):
        constraints = [constraints]

    wrapped_constraints = []
    for c in constraints:
        c_fun = c['fun']
        c_jac = c['jac']
        c_args = c.get('args', ())
        wrapped_constraints.append({
            'type': c['type'],
            'fun': (lambda x, f=c_fun, a=c_args: np.atleast_1d(f(x, *a))),
            'jac': (lambda x, j=c_jac, a=c_args: np.atleast_2d(j(x, *a))),
        })

    bounds_list = None
    if bounds is not None:
        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
            bounds_list = list(zip(bounds.lb, bounds.ub))
        else:
            bounds_list = list(bounds)
        bounds_list = [(None if lb == -np.inf else lb,
                         None if ub == np.inf else ub) for lb, ub in bounds_list]

    result = minimize_slp(fun_, np.asarray(x0, dtype=float), jac_,
                           constraints=wrapped_constraints, bounds=bounds_list,
                           maxiter=maxiter, tol=tol, delta0=delta0,
                           delta_min=delta_min, delta_max=delta_max,
                           eta1=eta1, eta2=eta2, shrink=shrink, expand=expand,
                           verbose=verbose)

    if callback is not None:
        for h in result.history:
            callback(h['x'])

    return OptimizeResult(
        x=result.x,
        fun=result.fun,
        jac=jac_(result.x),
        success=result.success,
        status=0 if result.success else 1,
        message=result.message,
        nit=result.nit,
        nfev=result.nit,
        njev=result.nit,
    )