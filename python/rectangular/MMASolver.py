"""
custom_mma.py
=============

Método de las Asíntotas Móviles (MMA, Svanberg 1987/2002) implementado
"a mano", con interfaz compatible con scipy.optimize.minimize (igual que
custom_sqp.py).

Arquitectura
------------
1. En cada iteración se construye una aproximación CONVEXA y SEPARABLE del
   problema original alrededor de x^(k):

        f0_tilde(x) = r0 + sum_j [ p0_j/(U_j - x_j) + q0_j/(x_j - L_j) ]
        fi_tilde(x) = ri + sum_j [ pi_j/(U_j - x_j) + qi_j/(x_j - L_j) ]

   con L_j, U_j asíntotas móviles que se actualizan según el historial de
   oscilación de cada variable (si una variable oscila, la asíntota se
   contrae; si avanza monótonamente, se expande). Esta parte -- asíntotas,
   límites de movimiento, coeficientes p/q -- es la lógica propia de MMA,
   implementada aquí desde cero con numpy.

2. El subproblema convexo separable se resuelve vía su DUAL, que es una
   maximización cóncava de dimensión pequeña (una variable y_i >= 0 por
   cada restricción). Dado y, el x que minimiza el Lagrangiano tiene forma
   cerrada (por separabilidad):

        x_j*(y) = clip( (sqrt(P_j(y))·L_j + sqrt(Q_j(y))·U_j)
                         / (sqrt(P_j(y)) + sqrt(Q_j(y))),  alpha_j, beta_j )

   y el gradiente del dual se obtiene por el teorema de la envolvente sin
   necesidad de diferenciar x*(y). Ese problema dual (pequeño, suave y
   cóncavo) se resuelve con scipy.optimize.minimize(method='L-BFGS-B')
   -- un primitivo numérico genérico, igual que usamos linprog en el SQP
   solo para una subtarea, no para resolver el problema no lineal original.

3. Convergencia por cambio en x y factibilidad de las restricciones.

Convención de restricciones (igual que scipy / custom_sqp):
    constraints = [
        {'type': 'ineq', 'fun': c, 'jac': dc},   # convención: c(x) >= 0
        {'type': 'eq',   'fun': c, 'jac': dc},   # se maneja partiéndola en
                                                   # dos desigualdades (nota
                                                   # abajo)
    ]

Nota importante: MMA fue diseñado para restricciones de DESIGUALDAD (es el
caso típico en optimización topológica: restricción de volumen, tensión
máxima, etc.). Las restricciones de igualdad se soportan partiéndolas en
c(x)>=0 y -c(x)>=0, pero al estar ambas activas simultáneamente en el
óptimo, la convergencia puede ser más lenta/oscilante que con un SQP. Si tu
problema es principalmente de igualdad, el SQP (custom_sqp.py) es más
apropiado.

bounds es OBLIGATORIO: las asíntotas y límites de movimiento de MMA se
definen relativos a (xmax - xmin), por lo que se requieren cotas finitas
en todas las variables.
"""

import numpy as np
from scipy.optimize import minimize as _scipy_minimize
import warnings


class MMAResult:
    """Contenedor de resultados, similar a scipy.optimize.OptimizeResult."""
    def __init__(self, x, fun, success, message, nit, history):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nit = nit
        self.history = history

    def __repr__(self):
        return (f"MMAResult(success={self.success}, fun={self.fun:.8g}, "
                f"nit={self.nit}, message='{self.message}')")


# ----------------------------------------------------------------------
# Actualización de asíntotas móviles y límites de movimiento
# ----------------------------------------------------------------------
def _update_asymptotes(it, x, xold1, xold2, xmin, xmax, low, upp,
                        asyinit, asyincr, asydecr):
    xrange = xmax - xmin
    xrange = np.maximum(xrange, 1e-10)

    if it <= 2:
        low = x - asyinit * xrange
        upp = x + asyinit * xrange
    else:
        zzz = (x - xold1) * (xold1 - xold2)
        gamma = np.ones_like(x)
        gamma[zzz > 0] = asyincr   # avance monótono -> expandir
        gamma[zzz < 0] = asydecr   # oscilación -> contraer
        low = x - gamma * (xold1 - low)
        upp = x + gamma * (upp - xold1)

    # Evitar asíntotas demasiado cercanas o demasiado lejanas
    low = np.maximum(low, x - 10 * xrange)
    low = np.minimum(low, x - 0.01 * xrange)
    upp = np.minimum(upp, x + 10 * xrange)
    upp = np.maximum(upp, x + 0.01 * xrange)
    return low, upp


def _move_limits(x, xmin, xmax, low, upp, move):
    xrange = xmax - xmin
    alpha = np.maximum(xmin, np.maximum(low + 0.1 * (x - low), x - move * xrange))
    beta = np.minimum(xmax, np.minimum(upp - 0.1 * (upp - x), x + move * xrange))
    return alpha, beta


# ----------------------------------------------------------------------
# Coeficientes p, q, r de la aproximación convexa separable
# ----------------------------------------------------------------------
def _approx_coeffs(f_val, df_dx, x, low, upp, xmin, xmax, raa):
    """
    f_val: escalar o array (m,)   -- f0(x) o [f1(x)...fm(x)]
    df_dx: array (n,) o (m,n)     -- gradiente(s)
    Devuelve p, q (misma forma que df_dx) y r (misma forma que f_val).
    """
    ux1 = upp - x
    xl1 = x - low
    xrange = np.maximum(xmax - xmin, 1e-10)

    pos = np.maximum(df_dx, 0.0)
    neg = np.maximum(-df_dx, 0.0)

    if df_dx.ndim == 1:
        p = ux1**2 * (1.001 * pos + 0.001 * neg + raa / xrange)
        q = xl1**2 * (0.001 * pos + 1.001 * neg + raa / xrange)
        r = f_val - np.sum(p / ux1 + q / xl1)
    else:
        p = ux1[None, :]**2 * (1.001 * pos + 0.001 * neg + raa / xrange[None, :])
        q = xl1[None, :]**2 * (0.001 * pos + 1.001 * neg + raa / xrange[None, :])
        r = f_val - np.sum(p / ux1[None, :] + q / xl1[None, :], axis=1)
    return p, q, r


def _primal_from_y(y, p0, q0, P, Q, low, upp, alpha, beta):
    """x*(y) en forma cerrada, dado el vector dual y (>=0). P,Q: (m,n)."""
    if y.size == 0:
        Pj, Qj = p0, q0
    else:
        Pj = p0 + y @ P
        Qj = q0 + y @ Q
    Pj = np.maximum(Pj, 1e-12)
    Qj = np.maximum(Qj, 1e-12)
    sqP, sqQ = np.sqrt(Pj), np.sqrt(Qj)
    x = (sqP * low + sqQ * upp) / (sqP + sqQ)
    return np.clip(x, alpha, beta)


def _solve_mma_subproblem(p0, q0, r0, P, Q, r, low, upp, alpha, beta):
    """
    Resuelve el subproblema convexo separable de MMA vía su dual.
    P, Q, r: coeficientes de las m restricciones (P,Q de forma (m,n), r de forma (m,)).
    Devuelve (x*, y*).
    """
    m = r.size
    if m == 0:
        x_star = _primal_from_y(np.zeros(0), p0, q0, np.zeros((0, len(p0))),
                                 np.zeros((0, len(p0))), low, upp, alpha, beta)
        return x_star, np.zeros(0)

    def neg_dual_and_grad(y):
        y = np.maximum(y, 0.0)
        x = _primal_from_y(y, p0, q0, P, Q, low, upp, alpha, beta)
        ux1 = upp - x
        xl1 = x - low
        f0_tilde = r0 + np.sum(p0 / ux1 + q0 / xl1)
        fi_tilde = r + np.sum(P / ux1[None, :] + Q / xl1[None, :], axis=1)
        w = f0_tilde + y @ fi_tilde
        grad = fi_tilde
        return -w, -grad

    y0 = np.zeros(m)
    res = _scipy_minimize(neg_dual_and_grad, y0, jac=True,
                           bounds=[(0, None)] * m, method='L-BFGS-B')
    y_star = np.maximum(res.x, 0.0)
    x_star = _primal_from_y(y_star, p0, q0, P, Q, low, upp, alpha, beta)
    return x_star, y_star


# ----------------------------------------------------------------------
# Solver principal
# ----------------------------------------------------------------------
def minimize_mma(fun, x0, jac, constraints=None, bounds=None,
                  maxiter=200, tol=1e-4, move=0.2,
                  asyinit=0.5, asyincr=1.2, asydecr=0.7,
                  raa0=1e-5, raa=1e-5, verbose=False):
    """
    Minimiza fun(x) sujeto a restricciones (principalmente de desigualdad)
    y bounds, usando el método de las Asíntotas Móviles (MMA).

    Parameters
    ----------
    fun : callable(x) -> float
    x0  : array_like
    jac : callable(x) -> array   (gradiente analítico, obligatorio)
    constraints : list of dict {'type': 'ineq'|'eq', 'fun':.., 'jac':..}
        Convención 'ineq': fun(x) >= 0 (igual que scipy/custom_sqp).
    bounds : list of (lb, ub), OBLIGATORIO (finitos) para todas las variables.
    maxiter, tol : criterios de paro (cambio en x y factibilidad).
    move : fracción máxima de (xmax-xmin) que puede moverse una variable
        por iteración (típico 0.1 - 0.3 en optimización topológica).
    asyinit, asyincr, asydecr : parámetros de la heurística de asíntotas.
    raa0, raa : regularización pequeña para evitar coeficientes p/q nulos.
    verbose : bool

    Returns
    -------
    MMAResult
    """
    if bounds is None:
        raise ValueError("MMA requiere 'bounds' finitos en todas las variables "
                          "(las asíntotas se definen relativas a xmax-xmin).")

    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    xmin = np.array([b[0] if b[0] is not None else -1e6 for b in bounds], dtype=float)
    xmax = np.array([b[1] if b[1] is not None else 1e6 for b in bounds], dtype=float)
    x = np.clip(x, xmin, xmax)

    constraints = constraints or []

    def eval_ineq(x):
        """Devuelve fi(x) <= 0 y su jacobiano, convirtiendo desde la
        convención scipy (fun(x) >= 0) y partiendo las 'eq' en dos ineq."""
        vals, jacs = [], []
        for c in constraints:
            v = np.atleast_1d(np.asarray(c['fun'](x), dtype=float))
            J = np.atleast_2d(np.asarray(c['jac'](x), dtype=float))
            if J.shape[0] != v.shape[0]:
                J = J.reshape(v.shape[0], -1)
            if c['type'] == 'ineq':
                vals.append(-v); jacs.append(-J)          # c>=0  ->  -c<=0
            elif c['type'] == 'eq':
                vals.append(-v); jacs.append(-J)          # c>=0 part
                vals.append(v); jacs.append(J)            # -c>=0 part (c<=0)
            else:
                raise ValueError(f"Tipo de restricción desconocido: {c['type']}")
        if not vals:
            return np.zeros(0), np.zeros((0, n))
        return np.concatenate(vals), np.vstack(jacs)

    low = xmin.copy()
    upp = xmax.copy()
    xold1 = x.copy()
    xold2 = x.copy()

    history = []
    success = False
    message = "Máximo de iteraciones alcanzado"
    it = 0

    for it in range(1, maxiter + 1):
        f0 = fun(x)
        df0 = np.asarray(jac(x), dtype=float)
        fval, dfdx = eval_ineq(x)   # fval <= 0 al converger
        m = fval.size

        low, upp = _update_asymptotes(it, x, xold1, xold2, xmin, xmax, low, upp,
                                       asyinit, asyincr, asydecr)
        alpha, beta = _move_limits(x, xmin, xmax, low, upp, move)

        p0, q0, r0 = _approx_coeffs(f0, df0, x, low, upp, xmin, xmax, raa0)
        if m > 0:
            P, Q, r = _approx_coeffs(fval, dfdx, x, low, upp, xmin, xmax, raa)
        else:
            P, Q, r = np.zeros((0, n)), np.zeros((0, n)), np.zeros(0)

        x_new, y = _solve_mma_subproblem(p0, q0, r0, P, Q, r, low, upp, alpha, beta)

        change = np.max(np.abs(x_new - x))
        feas = 0.0 if m == 0 else max(0.0, np.max(fval))

        if verbose:
            print(f"it={it:3d}  f={f0:.8g}  |dx|_inf={change:.3e}  feas={feas:.3e}")

        history.append(dict(it=it, f=f0, x=x.copy(), change=change, feas=feas))

        xold2 = xold1
        xold1 = x
        x = x_new

        if change < tol and feas < tol:
            success = True
            message = "Convergencia (cambio en x y factibilidad por debajo de tol)"
            break

    return MMAResult(x=x, fun=fun(x), success=success, message=message,
                      nit=it, history=history)


# ----------------------------------------------------------------------
# Adaptador para usarlo como method= en scipy.optimize.minimize
# ----------------------------------------------------------------------
def mma_method(fun, x0, args=(), jac=None, bounds=None, constraints=(),
               callback=None, maxiter=200, tol=1e-4, move=0.2,
               asyinit=0.5, asyincr=1.2, asydecr=0.7, verbose=False,
               **unused_options):
    """
    Adaptador MMA compatible con la interfaz de "custom minimizer" de scipy:

        from scipy.optimize import minimize
        from custom_mma import mma_method

        res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
                        method=mma_method,
                        options={'maxiter': 200, 'tol': 1e-4, 'move': 0.2})

    bounds es obligatorio (ver minimize_mma). Soporta args y callback igual
    que sqp_method en custom_sqp.py.
    """
    from scipy.optimize import OptimizeResult

    if jac is None:
        raise ValueError("mma_method requiere 'jac' (gradiente analítico).")
    if bounds is None:
        raise ValueError("mma_method requiere 'bounds' finitos (ver minimize_mma).")

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

    if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
        bounds_list = list(zip(bounds.lb, bounds.ub))
    else:
        bounds_list = list(bounds)
    bounds_list = [(None if lb == -np.inf else lb,
                     None if ub == np.inf else ub) for lb, ub in bounds_list]

    result = minimize_mma(fun_, np.asarray(x0, dtype=float), jac_,
                          constraints=wrapped_constraints, bounds=bounds_list,
                          maxiter=maxiter, tol=tol, move=move,
                          asyinit=asyinit, asyincr=asyincr, asydecr=asydecr,
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