"""GPU-only dense bounded TRF, adapted from SciPy 1.17.1.

Only the alpha solver's exact-SVD, linear-loss, unit-scale path is retained.
Source: scipy/optimize/_lsq/{trf,common}.py and optimize/_numdiff.py.
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
Distributed under the BSD-3-Clause license in licenses/scipy.txt.
"""
from types import SimpleNamespace as OptimizeResult
from math import copysign
import cupy as np
from cupy.linalg import norm, svd
EPS = np.finfo(float).eps


def difference(fun_many, x, f, lb, ub):
    """SciPy 1.17 two-point steps, batched without downloading residuals."""
    h = EPS**0.5 * np.where(x >= 0, 1., -1.) * np.maximum(1., np.abs(x))
    h, _ = _adjust_scheme_to_bounds(x, h, 1, '1-sided', lb, ub)
    trials = np.broadcast_to(x, (len(x), len(x))).copy()
    diagonal = np.arange(len(x))
    trials[diagonal, diagonal] += h
    dx = trials[diagonal, diagonal] - x
    return ((fun_many(trials) - f[None]) / dx[:, None]).T.copy()


def least_squares(fun_many, x0, bounds, *, max_nfev=500):
    """The project's fixed dense bounded TRF contract; arrays stay on CUDA."""
    lb, ub = (np.asarray(v, dtype=np.float64) for v in bounds)
    x = np.asarray(x0, dtype=np.float64)
    if (x.ndim != 1 or lb.shape != x.shape or ub.shape != x.shape
            or bool(np.any(lb >= ub)) or not bool(in_bounds(x, lb, ub))
            or not bool(np.isfinite(x).all()) or max_nfev < 1):
        raise ValueError('Invalid bounded alpha optimization inputs')
    x = make_strictly_feasible(x, lb, ub)
    latest_x = latest_f = None

    def fun(value):
        nonlocal latest_x, latest_f
        latest_x, latest_f = value.copy(), fun_many(value[None])[0]
        return latest_f

    def jac(value):
        base = latest_f if bool(np.array_equal(value, latest_x)) else fun(value)
        return difference(fun_many, value, base, lb, ub)

    f = fun(x)
    if not bool(np.isfinite(f).all()):
        raise ValueError('Non-finite initial alpha residual')
    J = jac(x)
    result = trf_bounds(fun, jac, x, f, J, lb, ub, 1e-8, 1e-8, 1e-8,
                        max_nfev, np.ones_like(x), None, 'exact', {}, 0)
    result.success = result.status > 0
    result.message = {0: 'The maximum number of function evaluations is exceeded.',
                      1: '`gtol` termination condition is satisfied.',
                      2: '`ftol` termination condition is satisfied.',
                      3: '`xtol` termination condition is satisfied.',
                      4: 'Both `ftol` and `xtol` termination conditions are satisfied.'}[result.status]
    return result

def intersect_trust_region(x, s, Delta):
    a = np.dot(s, s)
    if a == 0:
        raise ValueError('`s` is zero.')
    b = np.dot(x, s)
    c = np.dot(x, x) - Delta ** 2
    if c > 0:
        raise ValueError('`x` is not within the trust region.')
    d = np.sqrt(b * b - a * c)
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q
    if t1 < t2:
        return (t1, t2)
    else:
        return (t2, t1)

def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None, rtol=0.01, max_iter=10):

    def phi_and_derivative(alpha, suf, s, Delta):
        denom = s ** 2 + alpha
        p_norm = norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -np.sum(suf ** 2 / denom ** 3) / p_norm
        return (phi, phi_prime)
    suf = s * uf
    if m >= n:
        threshold = EPS * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False
    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return (p, 0.0, 0)
    alpha_upper = norm(suf) / Delta
    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0
    if initial_alpha is None or (not full_rank and initial_alpha == 0):
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    else:
        alpha = initial_alpha
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)
        if phi < 0:
            alpha_upper = alpha
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta
        if np.abs(phi) < rtol * Delta:
            break
    p = -V.dot(suf / (s ** 2 + alpha))
    p *= Delta / norm(p)
    return (p, alpha, it + 1)

def update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit):
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0
    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0
    return (Delta, ratio)

def build_quadratic_1d(J, g, s, diag=None, s0=None):
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return (a, b, c)
    else:
        return (a, b)

def minimize_quadratic_1d(a, b, lb, ub, c=0):
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.stack([np.asarray(value) for value in t])
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return (t[min_index], y[min_index])

def evaluate_quadratic(J, g, s, diag=None):
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.dot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js ** 2, axis=0)
        if diag is not None:
            q += np.sum(diag * s ** 2, axis=1)
    l = np.dot(s, g)
    return 0.5 * q + l

def step_size_to_bound(x, s, lb, ub):
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero, (ub - x)[non_zero] / s_non_zero)
    min_step = np.min(steps)
    return (min_step, np.equal(steps, min_step) * np.sign(s).astype(int))

def find_active_constraints(x, lb, ub, rtol=1e-10):
    active = np.zeros_like(x, dtype=int)
    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active
    lower_dist = x - lb
    upper_dist = ub - x
    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))
    lower_active = np.isfinite(lb) & (lower_dist <= np.minimum(upper_dist, lower_threshold))
    active[lower_active] = -1
    upper_active = np.isfinite(ub) & (upper_dist <= np.minimum(lower_dist, upper_threshold))
    active[upper_active] = 1
    return active

def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    x_new = x.copy()
    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)
    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        x_new[lower_mask] = lb[lower_mask] + rstep * np.maximum(1, np.abs(lb[lower_mask]))
        x_new[upper_mask] = ub[upper_mask] - rstep * np.maximum(1, np.abs(ub[upper_mask]))
    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])
    return x_new

def CL_scaling_vector(x, g, lb, ub):
    v = np.ones_like(x)
    dv = np.zeros_like(x)
    mask = (g < 0) & np.isfinite(ub)
    v[mask] = ub[mask] - x[mask]
    dv[mask] = -1
    mask = (g > 0) & np.isfinite(lb)
    v[mask] = x[mask] - lb[mask]
    dv[mask] = 1
    return (v, dv)

def compute_grad(J, f):
    return J.T.dot(f)

def in_bounds(x, lb, ub):
    return np.all((x >= lb) & (x <= ub))

def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)
    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None

def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return (p, p_h, -p_value)
    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
    ag_h = -g_h
    ag = d * ag_h
    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr
    a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride
    if p_value < r_value and p_value < ag_value:
        return (p, p_h, -p_value)
    elif r_value < p_value and r_value < ag_value:
        return (r, r_h, -r_value)
    else:
        return (ag, ag_h, -ag_value)

def trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose, callback=None):
    x = x0.copy()
    f = f0
    f_true = f.copy()
    nfev = 1
    J = J0
    njev = 1
    m, n = J.shape
    cost = 0.5 * np.dot(f, f)
    g = compute_grad(J, f)
    scale, scale_inv = (x_scale, 1 / x_scale)
    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v ** 0.5)
    if Delta == 0:
        Delta = 1.0
    g_norm = norm(g * v, ord=np.inf)
    f_augmented = np.zeros(m + n)
    J_augmented = np.empty((m + n, n))
    if max_nfev is None:
        max_nfev = x0.size * 100
    alpha = 0.0
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1
        if termination_status is not None or nfev == max_nfev:
            break
        v[dv != 0] *= scale_inv[dv != 0]
        d = v ** 0.5 * scale
        diag_h = g * dv * scale
        g_h = d * g
        f_augmented[:m] = f
        J_augmented[:m] = J * d
        J_h = J_augmented[:m]
        J_augmented[m:] = np.diag(diag_h ** 0.5)
        U, s, V = svd(J_augmented, full_matrices=False)
        V = V.T
        uf = U.T.dot(f_augmented)
        theta = max(0.995, 1 - g_norm)
        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)
            p = d * p_h
            step, step_h, predicted_reduction = select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)
            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1
            step_h_norm = norm(step_h)
            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > 0.95 * Delta)
            step_norm = norm(step)
            termination_status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break
            alpha *= Delta / Delta_new
            Delta = Delta_new
        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.copy()
            cost = cost_new
            J = jac(x)
            njev += 1
            g = compute_grad(J, f)
        else:
            step_norm = 0
            actual_reduction = 0
        iteration += 1
    if termination_status is None:
        termination_status = 0
    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm, active_mask=active_mask, nfev=nfev, njev=njev, status=termination_status)

def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
    if scheme == '1-sided':
        use_one_sided = np.ones_like(h, dtype=bool)
    elif scheme == '2-sided':
        h = np.abs(h)
        use_one_sided = np.zeros_like(h, dtype=bool)
    else:
        raise ValueError("`scheme` must be '1-sided' or '2-sided'.")
    if np.all((lb == -np.inf) & (ub == np.inf)):
        return (h, use_one_sided)
    h_total = h * num_steps
    h_adjusted = h.copy()
    lower_dist = x0 - lb
    upper_dist = ub - x0
    if scheme == '1-sided':
        x = x0 + h_total
        violated = (x < lb) | (x > ub)
        fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
        h_adjusted[violated & fitting] *= -1
        forward = (upper_dist >= lower_dist) & ~fitting
        h_adjusted[forward] = upper_dist[forward] / num_steps
        backward = (upper_dist < lower_dist) & ~fitting
        h_adjusted[backward] = -lower_dist[backward] / num_steps
    elif scheme == '2-sided':
        central = (lower_dist >= h_total) & (upper_dist >= h_total)
        forward = (upper_dist >= lower_dist) & ~central
        h_adjusted[forward] = np.minimum(h[forward], 0.5 * upper_dist[forward] / num_steps)
        use_one_sided[forward] = True
        backward = (upper_dist < lower_dist) & ~central
        h_adjusted[backward] = -np.minimum(h[backward], 0.5 * lower_dist[backward] / num_steps)
        use_one_sided[backward] = True
        min_dist = np.minimum(upper_dist, lower_dist) / num_steps
        adjusted_central = ~central & (np.abs(h_adjusted) <= min_dist)
        h_adjusted[adjusted_central] = min_dist[adjusted_central]
        use_one_sided[adjusted_central] = False
    return (h_adjusted, use_one_sided)
