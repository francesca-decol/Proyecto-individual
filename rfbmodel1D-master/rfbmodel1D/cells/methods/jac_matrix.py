"""
Robust Jacobian-sparsity construction for the full-cell DAE.

Paste these methods into the `Cell` class in cell.py (they use `self`,
`self.dae_residual_full_cell`, `self.initialize_dae_state`, and the
layout/mesh attributes already present on Cell).

Why multi-state:
    The residual contains several data-dependent branches that change WHICH
    state entries couple into a given equation, not just their values:
      * membrane EOD upwind:        `if i_io > 0` selects lam[:-2] vs lam[2:]
      * sorption / desorption:      hard `if` and tanh-blends at delta_lam ~ 0
      * convection scheme select:   np.where(Pe < 2, central, upwind)
      * condensation / evaporation: `if c_vw > c_sat`
    A single-state finite-difference probe only sees the branch active at that
    state, so it can omit real couplings. We therefore OR the detected pattern
    over several representative states spanning charge AND discharge, plus
    jittered copies that push the FD stencil onto both sides of each switch.

Safety direction:
    A sparsity pattern that is too DENSE is safe: the solver simply probes a
    few entries that turn out to be zero (a speed cost, never a correctness
    cost). A pattern that is too SPARSE is the bug we are fixing: the solver
    treats a real coupling as structurally zero and never recovers it. Every
    choice here errs toward over-inclusion.
"""
import numpy as np
from scipy.optimize import approx_fprime
from scipy.sparse import csr_matrix


def _single_state_pattern(self, y, yp, t, eps, tol):
    """Boolean (dense) nonzero patterns of dF/dy and dF/dyp at one state."""
    res = lambda yy, ypp: self.dae_residual_full_cell(t, yy, ypp)
    J_y  = approx_fprime(y.copy(),  lambda yy:  res(yy, yp), eps)
    J_yp = approx_fprime(yp.copy(), lambda ypp: res(y, ypp), eps)
    return (np.abs(J_y) > tol), (np.abs(J_yp) > tol) #type: ignore


def _representative_states(self, base_tanks=None, n_jitter=2, jitter=0.05, seed=0):
    """
    Yield (y, yp, label) tuples spanning charge and discharge, plus jittered
    copies. Uses initialize_dae_state to get DAE-consistent vectors, then
    perturbs to land on both sides of the internal switches.

    base_tanks: list of c_tank arrays [c_V4, c_V5, c_H, c_HSO4, V_T] (mol/m3, m3).
                If None, a low-SOC and a high-SOC tank are used so that both
                charge-like and discharge-like operating points are covered.
    """
    rng = np.random.default_rng(seed)

    if base_tanks is None:
        # Two SOC extremes. SOC = c_V5 / (c_V4 + c_V5); the sign of the working
        # current (and hence i_io, and hence the EOD upwind branch) differs
        # between a nearly-charged and nearly-discharged tank under the same
        # applied BC, so this pair exercises both i_io signs.
        V_T = 1.0e-4
        c_H, c_HSO4 = 6500.0, 4000.0
        base_tanks = [
            np.array([900.0,  100.0, c_H, c_HSO4, V_T]),  # low SOC  (discharge-leaning)
            np.array([100.0,  900.0, c_H, c_HSO4, V_T]),  # high SOC (charge-leaning)
        ]

    # Both BC modes flip the cathode potential boundary condition handling,
    # which routes through different branches in update_BC_cathode_membrane.
    original_mode = self.bc_mode
    modes = ["discharge", "charge"]

    for tank in base_tanks:
        for mode in modes:
            # Map our conceptual mode onto the applied electronic-potential BC.
            # Discharge: lower cell potential pulls current one way;
            # charge: raise it to reverse i_io. We nudge the phi_el left BC.
            try:
                if mode == "charge":
                    self._set_charge_like_bc()
                else:
                    self._set_discharge_like_bc()
            except AttributeError:
                # Helpers optional; if absent we still get SOC-driven sign
                # variation from the two tanks above.
                pass

            try:
                y0, yp0 = self.initialize_dae_state(tank, xtol=1e-6)
            except Exception as e:
                # A non-converged init at an extreme tank is fine to skip;
                # the other states still cover the branches.
                print(f"[sparsity] init skipped (tank={tank[:2]}, {mode}): {e}")
                continue

            yield y0.copy(), yp0.copy(), f"{mode}_SOC{tank[1]/(tank[0]+tank[1]):.2f}"

            # Jittered copies: multiplicative noise on y, additive on yp.
            # This pushes delta_lam, (c_vw - c_sat), and Pe across their
            # switch points so both branch sides get sampled.
            for k in range(n_jitter):
                y_j  = y0 * (1.0 + jitter * rng.standard_normal(y0.shape))
                yp_j = yp0 + jitter * np.maximum(np.abs(yp0), 1e-12) \
                       * rng.standard_normal(yp0.shape)
                yield y_j, yp_j, f"{mode}_SOC{tank[1]/(tank[0]+tank[1]):.2f}_jit{k}"

    # Restore whatever mode the cell started in.
    self.bc_mode = original_mode


def build_jac_sparsity(self, t=0.0, eps=1e-7, tol=1e-30,
                       base_tanks=None, n_jitter=2, jitter=0.05, seed=0,
                       return_dense=False, verbose=True):
    """
    Build robust (sparsity_y, sparsity_yp) by OR-ing single-state FD patterns
    over representative charge/discharge states and jittered neighbours.

    Parameters
    ----------
    t : float
        Time at which to evaluate (residual is autonomous in practice).
    eps : float
        FD step for approx_fprime.
    tol : float
        Threshold above which an entry is counted nonzero. Keep this SMALL
        (default 1e-30, i.e. "any nonzero FD response"). Raising tol drops
        weak-but-real couplings and reintroduces the missing-entry bug; the
        only reason to raise it is to suppress pure round-off, and even then
        a too-dense pattern is the safe failure mode.
    base_tanks, n_jitter, jitter, seed :
        Passed to _representative_states.
    return_dense : bool
        If True, also return the dense boolean union arrays (Sy_dense, Syp_dense)
        so you can diff against a hand-built pattern.
    verbose : bool
        Print per-state nnz growth so you can confirm later states are still
        adding entries (i.e. that you covered enough branches).

    Returns
    -------
    (Sy, Syp)                       if return_dense is False
    (Sy, Syp, Sy_dense, Syp_dense)  if return_dense is True
    """
    n = None
    Sy = None
    Syp = None
    n_states = 0

    for y, yp, label in self._representative_states(
            base_tanks=base_tanks, n_jitter=n_jitter, jitter=jitter, seed=seed):
        py, pyp = self._single_state_pattern(y, yp, t, eps, tol)
        if Sy is None:
            n = y.size
            Sy  = np.zeros((n, n), dtype=bool)
            Syp = np.zeros((n, n), dtype=bool)
        Sy  |= py
        Syp |= pyp
        n_states += 1
        if verbose:
            print(f"[sparsity] state {n_states:2d} ({label}): "
                  f"nnz(dF/dy)={Sy.sum():6d}  nnz(dF/dyp)={Syp.sum():5d}")

    if Sy is None:
        raise RuntimeError(
            "No representative state initialized successfully; "
            "cannot build sparsity pattern.")

    # Diagonal insurance: every variable must at least appear in its own
    # equation. Costs nothing and guards against an unlucky all-zero column.
    np.fill_diagonal(Sy, True)
    # dF/dyp: only DIFFERENTIAL variables have a yp dependence. Forcing a full
    # diagonal here would mislabel algebraic equations as differential, so we
    # do NOT fill it — we trust the FD detection, which correctly leaves
    # algebraic rows empty in dF/dyp.

    if verbose:
        dens_y  = Sy.sum()  / Sy.size
        dens_yp = Syp.sum() / Syp.size #type: ignore
        print(f"[sparsity] final: n={n}, states={n_states}, "
              f"density(dF/dy)={dens_y:.4%}, density(dF/dyp)={dens_yp:.4%}")

    Sy_csr  = csr_matrix(Sy.astype(int))
    Syp_csr = csr_matrix(Syp.astype(int)) #type: ignore

    if return_dense:
        return Sy_csr, Syp_csr, Sy, Syp
    return Sy_csr, Syp_csr


def compare_to_pattern(self, candidate_csr, y=None, yp=None, t=0.0,
                       eps=1e-7, tol=1e-30, which="y", **build_kwargs):
    """
    Diff a candidate (e.g. hand-built) sparsity pattern against the robust
    dense union. Returns indices of entries that the robust pattern marks
    nonzero but the candidate marks zero -- i.e. the couplings the candidate
    is MISSING (the dangerous direction).

    which : "y" or "yp" -- which Jacobian's pattern to compare.
    """
    Sy_csr, Syp_csr, Sy_dense, Syp_dense = self.build_jac_sparsity(
        t=t, eps=eps, tol=tol, return_dense=True, verbose=False, **build_kwargs)
    robust = Sy_dense if which == "y" else Syp_dense
    cand = candidate_csr.toarray().astype(bool)
    if cand.shape != robust.shape:
        raise ValueError(f"shape mismatch: candidate {cand.shape} vs robust {robust.shape}")
    missing = robust & ~cand          # real coupling the candidate dropped
    extra   = cand & ~robust          # candidate marks nonzero, robust didn't (safe)
    missing_idx = np.argwhere(missing)
    print(f"[compare:{which}] candidate nnz={cand.sum()}, robust nnz={robust.sum()}")
    print(f"[compare:{which}] MISSING (dangerous) = {missing.sum()} entries")
    print(f"[compare:{which}] extra   (harmless)  = {extra.sum()} entries")
    if missing.sum():
        print(f"[compare:{which}] first missing (row=res idx, col=state idx):")
        print(missing_idx[:20])
    return missing_idx