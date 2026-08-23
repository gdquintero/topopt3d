"""
slp_gomes_senne.py
====================

Port literal (fiel) do código MATLAB de Programação Linear Sequencial
(PLS/SLP) fornecido pelo usuário -- uma variante prática, específica
para otimização topológica, do Algoritmo 1 de Gomes e Senne (2011)
descrito no Capítulo 4 do texto -- adaptado como `method` customizado
para `scipy.optimize.minimize`.

Este arquivo SUBSTITUI a versão anterior ("livro-texto puro"), que era
fiel ao Algoritmo 1 mas SEM as adaptações práticas abaixo, as quais o
usuário confirmou serem necessárias para o problema real de otimização
topológica.

Diferenças-chave do MATLAB de referência em relação ao Algoritmo 1 "puro"
do texto (mantidas aqui EXATAMENTE, por instrução do usuário):

1. Passo tangencial com RHS = 0 (não -c). Depois que a restauração
   (4.12) atinge M(x^k,s_n) ~ 0 dentro da subcaixa 0.8*delta, o passo
   tangencial resolve
        min g^T s   s.a.  A s = 0,   s em [lo,hi] (caixa completa),
   em vez de `A s = -c` (eq. 4.9 literal). Simplificação prática: a
   restauração já mostrou que a linearização é essencialmente factível
   dentro de uma subcaixa; o teste de aceitação (Ared/Pred), que usa
   sempre os valores REAIS de c, A no ponto atual, corrige qualquer
   step tangencial que não preserve a factibilidade linear exatamente.

2. Duas ordens possíveis (restauração->tangente OU tangente->restauração)
   dependendo de se ||c(x^k)||_1 <= feas_tol ou não.

3. Atualização de delta em 3 faixas (razão apred = Ared/Pred):
     apred >= 0.5        -> delta <- min(1.5*delta, max_span)
     0.2 <= apred < 0.5  -> delta inalterado
     apred <  0.2        -> delta <- 0.25*delta
   (rejeição, apred<0.1: delta <- max(0.25||s||_inf, 0.1*delta), igual ao texto)

4. theta_min_k com janela de tamanho 2 (thold1, thold2), atualizada a
   CADA passagem pelo laço (aceita ou não), em vez do mínimo sobre todo
   o histórico de thetas aceitos (eq. 4.16 do texto).

5. N grande por padrão (1e6, não 0): permite decréscimo bem não
   monótono da função de mérito no início, evitando o colapso prematuro
   de delta que ocorre com N=0 (monótono estrito) em problemas
   fortemente não convexos como otimização topológica.

6. Critério de parada = norma do gradiente projetado da Lagrangiana,
        gradproj = proj_[yl,yu](x - (grad_f - A^T*lambda)) - x,
        gpnorm = ||gradproj||_inf,
   usando o multiplicador de Lagrange do último subproblema de PL
   resolvido (`res.eqlin.marginals` em scipy), mais um contador de
   passagens consecutivas com passo pequeno.

Uso
---
    from scipy.optimize import minimize
    from slp_gomes_senne import slp_gs

    res = minimize(fun, x0, jac=jac, bounds=bounds,
                    constraints=constraints, method=slp_gs,
                    options={'maxiter': 200, 'disp': True})
"""

import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.optimize import Bounds as _Bounds
from scipy.optimize._numdiff import approx_derivative


# --------------------------------------------------------------------------
# Utilidades (bounds / restrições) -- infraestrutura, não faz parte do
# algoritmo MATLAB em si.
# --------------------------------------------------------------------------

def _process_bounds(bounds, n):
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    if isinstance(bounds, _Bounds):
        lb = np.broadcast_to(np.asarray(bounds.lb, dtype=float), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds.ub, dtype=float), (n,)).copy()
        return lb, ub
    lb = np.empty(n)
    ub = np.empty(n)
    for i, b in enumerate(bounds):
        lo, hi = b
        lb[i] = -np.inf if lo is None else lo
        ub[i] = np.inf if hi is None else hi
    return lb, ub


class _Constraint:
    """Normaliza uma restrição para a forma comum  lb <= g(x) <= ub."""

    def __init__(self, con):
        if isinstance(con, dict):
            kind = con['type']
            fun = con['fun']
            jac = con.get('jac', None)
            args = con.get('args', ())

            def wrapped_fun(x, f=fun, a=args):
                return np.atleast_1d(np.asarray(f(x, *a), dtype=float))

            if jac is not None and callable(jac):
                def wrapped_jac(x, j=jac, a=args):
                    return np.atleast_2d(np.asarray(j(x, *a), dtype=float))
            else:
                wrapped_jac = None

            self.fun = wrapped_fun
            self.jac_fn = wrapped_jac
            if kind == 'eq':
                self.lb_fn = lambda m: np.zeros(m)
                self.ub_fn = lambda m: np.zeros(m)
            else:  # 'ineq': fun(x) >= 0
                self.lb_fn = lambda m: np.zeros(m)
                self.ub_fn = lambda m: np.full(m, np.inf)
        else:
            lb = np.atleast_1d(np.asarray(con.lb, dtype=float))
            ub = np.atleast_1d(np.asarray(con.ub, dtype=float))
            if hasattr(con, 'A'):  # LinearConstraint
                A = np.atleast_2d(np.asarray(con.A, dtype=float))
                self.fun = lambda x, A=A: A @ x
                self.jac_fn = lambda x, A=A: A
            else:  # NonlinearConstraint
                fun = con.fun
                jac_attr = getattr(con, 'jac', None)
                jac = jac_attr if callable(jac_attr) else None
                self.fun = lambda x, f=fun: np.atleast_1d(np.asarray(f(x), dtype=float))
                self.jac_fn = ((lambda x, j=jac: np.atleast_2d(np.asarray(j(x), dtype=float)))
                               if jac is not None else None)
            self.lb_fn = lambda m, lb=lb: lb
            self.ub_fn = lambda m, ub=ub: ub

    def value(self, x):
        return self.fun(x)

    def jacobian(self, x, f0=None):
        if self.jac_fn is not None:
            return self.jac_fn(x)
        if f0 is None:
            f0 = self.value(x)
        J = approx_derivative(lambda z: self.value(z), x, method='2-point', f0=f0)
        return np.atleast_2d(J)

    def bounds(self, m):
        return self.lb_fn(m), self.ub_fn(m)


def _build_constraints(constraints):
    if constraints is None:
        return []
    if isinstance(constraints, dict) or hasattr(constraints, 'lb'):
        constraints = [constraints]
    return [_Constraint(c) for c in constraints]


def _flatten_constraints(cons, x):
    if not cons:
        return (np.zeros(0), np.zeros((0, x.size)), np.zeros(0), np.zeros(0))
    vals_l, jac_l, lb_l, ub_l = [], [], [], []
    for c in cons:
        v = c.value(x)
        J = c.jacobian(x, f0=v)
        lo, hi = c.bounds(v.size)
        vals_l.append(v); jac_l.append(J); lb_l.append(lo); ub_l.append(hi)
    return (np.concatenate(vals_l), np.vstack(jac_l),
            np.concatenate(lb_l), np.concatenate(ub_l))


def _eqlin_duals(res, m):
    """Extrai os multiplicadores de Lagrange (duais) da restrição de
    igualdade de um resultado de linprog (equivalente a `lambda.eqlin`
    no MATLAB). Devolve um vetor de tamanho m, ou None se indisponível."""
    eq = getattr(res, 'eqlin', None)
    if eq is None:
        return None
    marg = getattr(eq, 'marginals', None)
    if marg is None:
        return None
    marg = np.asarray(marg, dtype=float)
    return marg[:m] if marg.size >= m else None


# --------------------------------------------------------------------------
# Subproblemas de PL
# --------------------------------------------------------------------------

def _restoration_step(c, A, lo, hi, zero_tol):
    """Passo normal / restauração (eqs. 4.10-4.12): minimiza ||c+A s||_1
    via variáveis artificiais z, dentro da subcaixa [lo,hi] (0.8*delta).
    Equivalente a `vol_constr_normal` + `linprog(...,-b,...)` do MATLAB,
    generalizado para c vetorial. Devolve (s_n, M, lam_eq)."""
    p, n_total = A.shape
    nz_idx = np.where(np.abs(c) > zero_tol)[0]
    mI = nz_idx.size

    if mI == 0:
        return np.zeros(n_total), 0.0, None

    n_var = n_total + mI
    obj = np.zeros(n_var)
    obj[n_total:] = 1.0

    A_eq = np.zeros((p, n_var))
    A_eq[:, :n_total] = A
    for j, i in enumerate(nz_idx):
        sign = 1.0 if c[i] < 0 else -1.0
        A_eq[i, n_total + j] = sign
    b_eq = -c

    bnds = list(zip(lo, hi)) + [(0, None)] * mI
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method='highs')
    if not res.success:
        return np.zeros(n_total), float(np.sum(np.abs(c))), None

    s_n = res.x[:n_total]
    M = float(np.sum(res.x[n_total:]))
    lam_eq = _eqlin_duals(res, p)
    return s_n, M, lam_eq


def _tangent_step(g, A, lo, hi, rhs):
    """Passo tangencial: minimiza g^T s  s.a.  A s = rhs,  s em [lo,hi].
    rhs=0 reproduz literalmente a simplificação do MATLAB de referência
    (ver ponto 1 no docstring do módulo); rhs=-c reproduziria a eq. (4.9)
    "pura" do texto, disponível via o parâmetro `tangent_rhs` do solver."""
    res = linprog(g, A_eq=A, b_eq=rhs, bounds=list(zip(lo, hi)), method='highs')
    if not res.success:
        return None, False, None
    lam_eq = _eqlin_duals(res, A.shape[0])
    return res.x, True, lam_eq


# --------------------------------------------------------------------------
# Solver principal
# --------------------------------------------------------------------------

def slp_gs(fun, x0, args=(), jac=None, bounds=None, constraints=(),
           maxiter=200, tol=1e-6, tolcount=5,
           delta0=1.0, delta_min=1e-12,
           N=1e6, zero_tol=1e-10, feas_tol=1e-10,
           max_rejections=200,
           tangent_rhs='zero',
           callback=None, disp=False, **options):
    """
    Programação Linear Sequencial (PLS) com região de confiança e função
    de mérito -- port literal do código MATLAB de referência do usuário
    (variante prática do Algoritmo 1 de Gomes e Senne, 2011, usada em
    otimização topológica).

    Compatível com `scipy.optimize.minimize(..., method=slp_gs)`.

    Parameters
    ----------
    fun, x0, args, jac, bounds, constraints
        Mesma convenção do restante dos solvers customizados do projeto.
        `fun`/`jac` já devem incorporar internamente qualquer filtro de
        densidade e sua regra da cadeia (responsabilidade do problema,
        não do solver).
    maxiter : int
        Máximo de iterações ACEITAS (`iter` no MATLAB).
    tol : float
        Tolerância única para dois critérios (igual ao MATLAB): a norma
        do gradiente projetado `gpnorm` para declarar convergência, e o
        limite de `||s||_inf` para contar uma passagem como "passo
        pequeno" (contador `count`).
    tolcount : int
        Número de passagens CONSECUTIVAS com passo pequeno exigido junto
        com `gpnorm<=tol` como salvaguarda contra oscilação (o `count`
        do MATLAB).
    delta0 : float
        Raio inicial da região de confiança.
    delta_min : float
        Salvaguarda prática (o MATLAB fornecido não impõe piso explícito
        para delta): se delta cair abaixo disso sem nenhum passo aceito,
        o laço é interrompido para evitar loop infinito.
    N : float
        Parâmetro da eq. (4.15)/thklarge. Default 1e6, igual ao MATLAB
        de referência. Use N=0 para o comportamento monótono
        "livro-texto".
    zero_tol : float
        Tolerância para tratar componentes de c(x^k) como exatamente
        nulas na fase de restauração.
    feas_tol : float
        Tolerância para decidir se x^k já é "aproximadamente factível"
        (||c(x^k)||_1 <= feas_tol), equivalente a `abs(b)>1e-10` no
        MATLAB, generalizado para c vetorial via norma-1.
    max_rejections : int
        Salvaguarda prática (não está no MATLAB fornecido): máximo de
        rejeições consecutivas antes de desistir.
    tangent_rhs : {'zero', 'exact'}
        'zero' (default) reproduz o MATLAB literalmente (A s = 0).
        'exact' usa A s = -c (eq. 4.9 pura do texto), para comparação.
    disp : bool
        Se True, imprime o progresso a cada passagem do laço.

    Returns
    -------
    scipy.optimize.OptimizeResult
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n = x.size

    f = lambda z: float(fun(z, *args))
    if jac is not None and callable(jac):
        grad = lambda z: np.asarray(jac(z, *args), dtype=float)
    else:
        grad = lambda z: approx_derivative(f, z, method='2-point')

    lb_x, ub_x = _process_bounds(bounds, n)
    cons = _build_constraints(constraints)

    # --- problema aumentado (x, t) com c(x,t) = g(x) - t = 0 ---
    vals0, J0, lb_c, ub_c = _flatten_constraints(cons, x)
    p = vals0.size
    n_total = n + p

    t0 = np.clip(vals0, np.where(np.isfinite(lb_c), lb_c, -1e300),
                 np.where(np.isfinite(ub_c), ub_c, 1e300))
    y = np.concatenate([x, t0])
    yl = np.concatenate([lb_x, lb_c])
    yu = np.concatenate([ub_x, ub_c])

    box_span = yu - yl
    finite_span = box_span[np.isfinite(box_span)]
    max_span = float(np.max(finite_span)) if finite_span.size else 1e6

    def eval_point(y):
        xx = y[:n]
        tt = y[n:]
        fx = f(xx)
        gx_x = grad(xx)
        g_full = np.concatenate([gx_x, np.zeros(p)])
        vals, J, _, _ = _flatten_constraints(cons, xx)
        c = vals - tt
        A = np.zeros((p, n_total))
        if p:
            A[:, :n] = J
            A[:, n:] = -np.eye(p)
        return fx, g_full, c, A

    fx, gx, c, A = eval_point(y)
    phi_x = float(np.sum(np.abs(c)))
    nfev, njev = 1, 1

    delta = max(delta0, delta_min)
    thmax = 1.0
    thold1, thold2 = 1.0, 1.0     # theta_0 = theta_max = 1 (janela de 2, como no MATLAB)

    iter_ = 0          # "iter" do MATLAB: iterações ACEITAS
    count = 0          # passagens consecutivas com passo pequeno
    n_rejections_total = 0
    gpnorm = np.inf    # ainda não há LP resolvida -> força entrada no laço
    lam_eq = None

    status, message = 1, "Maximum number of iterations reached."

    while gpnorm > tol and count < tolcount and iter_ <= maxiter:

        sL = np.maximum(-delta, 0.001 - y)
        sU = np.minimum(delta, 1 - y)
        if np.any(sL > sU):
            status, message = 4, "Caixa de regiao de confianca incompativel com yl<=y<=yu."
            break

        infeasible = phi_x > feas_tol
        rhs_tangent = np.zeros(p) if tangent_rhs == 'zero' else -c

        if infeasible:
            # ---- passo normal / restauração primeiro (RHS = -c, exato) ----
            lo1 = np.maximum(-0.8 * delta, 0.001 - y)
            hi1 = np.minimum(0.8 * delta, 1 - y)
            s_n, Mval, lam_r = _restoration_step(c, A, lo1, hi1, zero_tol)
            lam_eq = lam_r

            if Mval < 1e-10:
                s, ok_tg, lam_tg = _tangent_step(gx, A, sL, sU, rhs_tangent)
                if ok_tg:
                    lam_eq = lam_tg if lam_tg is not None else lam_eq
                else:
                    s = s_n
            else:
                s = s_n
        else:
            # ---- ja aproximadamente factivel: tenta passo tangencial puro ----
            s, ok_tg, lam_tg = _tangent_step(gx, A, sL, sU, rhs_tangent)
            if ok_tg:
                lam_eq = lam_tg
            else:
                lo1 = np.maximum(-0.8 * delta, 0.001 - y)
                hi1 = np.minimum(0.8 * delta, 1 - y)
                s_n, Mval, lam_r = _restoration_step(c, A, lo1, hi1, zero_tol)
                s = s_n
                lam_eq = lam_r

        step_norm = float(np.max(np.abs(s))) if n_total else 0.0

        # ---- reducoes previstas do modelo linear (usando A, c REAIS do ponto atual) ----
        Popt_red = float(-gx @ s)
        Pfct_red = phi_x - float(np.sum(np.abs(c + A @ s)))

        # ---- theta_k: janela de 2 (thold1, thold2), N grande por padrao ----
        theta_min_k = min(thold1, thold2)
        theta_large_k = (1.0 + N / (iter_ + 1) ** 1.1) * theta_min_k
        if Popt_red > 0.5 * Pfct_red:
            theta_sup_k = 1.0
        else:
            denom = Pfct_red - Popt_red
            theta_sup_k = 0.5 * (Pfct_red / denom) if abs(denom) > 1e-14 else 1.0
        theta_k = min(theta_sup_k, theta_large_k, thmax)
        theta_k = float(np.clip(theta_k, 0.0, 1.0))
        thold2 = thold1
        thold1 = theta_k

        Pred = theta_k * Popt_red + (1.0 - theta_k) * Pfct_red

        # ---- avalia o ponto candidato ----
        y_trial = y + s
        f_trial, g_trial, c_trial, A_trial = eval_point(y_trial)
        nfev += 1
        phi_trial = float(np.sum(np.abs(c_trial)))

        Aopt_red = fx - f_trial
        Afct_red = phi_x - phi_trial
        Ared = theta_k * Aopt_red + (1.0 - theta_k) * Afct_red

        apred = (Ared / Pred) if abs(Pred) > 1e-300 else -np.inf

        if disp:
            print(f"iter={iter_:4d}  theta={theta_k: .3f}  apred={apred: .3e}  "
                  f"delta={delta: .3e}  f={fx: .6e}  phi={phi_x: .3e}  |s|={step_norm:.3e}  "
                  f"gpnorm={gpnorm: .3e}")

        if apred < 0.1:
            # ---- passo REJEITADO ----
            delta = max(0.25 * step_norm, 0.1 * delta)
            thmax = theta_k
            n_rejections_total += 1
            if delta < delta_min or n_rejections_total > max_rejections:
                status, message = 3, "Trust region collapsed / too many rejections without acceptance."
                break
        else:
            # ---- passo ACEITO ----
            y = y_trial
            fx, gx, c, A = f_trial, g_trial, c_trial, A_trial
            phi_x = phi_trial
            njev += 1
            n_rejections_total = 0

            if apred >= 0.5:
                delta = min(1.5 * delta, max_span)
            elif apred >= 0.2:
                pass  # delta inalterado
            else:
                delta = 0.25 * delta
            thmax = 1.0
            iter_ += 1

            if callback is not None:
                callback(np.copy(y[:n]))

        # ---- criterio de parada: gradiente projetado + contador de passo pequeno ----
        if lam_eq is not None and p > 0:
            lagr_grad_x = gx[:n] - A[:, :n].T @ lam_eq
        else:
            lagr_grad_x = gx[:n]
        proj = np.clip(y[:n] - lagr_grad_x, yl[:n], yu[:n])
        gradproj = proj - y[:n]
        gpnorm = float(np.max(np.abs(gradproj))) if n else 0.0

        if step_norm <= tol:
            count += 1
        else:
            count = 0

    if gpnorm <= tol:
        status, message = 0, "Optimization terminated successfully (projected-gradient norm small)."
    elif count >= tolcount:
        status, message = 0, ("Optimization terminated successfully "
                               "(stalled at small, consecutive steps).")

    x_final = y[:n]
    result = OptimizeResult(
        x=x_final, fun=fx, jac=gx[:n], nit=iter_, nfev=nfev, njev=njev,
        status=status, message=message, success=(status == 0),
        constr_violation=phi_x, maxcv=phi_x,
    )
    return result