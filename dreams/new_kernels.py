import treams.special as sp
import jax.numpy as jnp
from scipy.special import gammaln


# def pure_tl_vsw_helper(l, m, lambda_, mu, p, q):
    
#     def callback(cur_l, cur_m, cur_lambda_, cur_mu, cur_p, cur_q):
#         return sp._tl_vsw_helper(l, m, lambda_, -mu, cur_p, cur_p)
    



def tl_vsw_helper_jax(l, m, lambda_, mu, p, q):
    """
    Vectorized NumPy version of the translation coefficient helper.
    """
    
    # 1. Cast inputs to arrays for consistent broadcasting

    # 2. Pre-calculate absolute differences/sums needed for selection rules
    abs_m_plus_mu = jnp.abs(m + mu)
    abs_l_minus_lam = jnp.abs(l - lambda_)
    abs_l_plus_lam = jnp.abs(l + lambda_)

    # 3. Define the Selection Rules Mask (The "Gatekeeper")
    # This corresponds to the big 'if' statement in the Cython code.
    # We identify indices where the result is effectively 0.
    
    # Triangle inequalities: |l - lambda| <= p <= |l + lambda|
    invalid_p = (p < jnp.maximum(abs_m_plus_mu, abs_l_minus_lam)) | \
                (p > abs_l_plus_lam)
                
    # Triangle inequalities for q
    invalid_q = (q < abs_l_minus_lam) | \
                (q > abs_l_plus_lam)
                
    # Parity check
    invalid_parity = (q + l + lambda_) % 2 != 0

    # Combine all invalid conditions
    zero_mask = invalid_p | invalid_q | invalid_parity

    # 4. Calculate the Terms
    # We perform calculations everywhere (even invalid spots) to keep vectorization fast,
    # then mask the zeros at the end.
    
    # Term: (2p + 1)
    term_normalization = 2 * p + 1

    # Term: i^(lambda - l + p)
    # Using 1j for the imaginary unit
    term_phase = (1j) ** (lambda_ - l + p)

    # Term: Sqrt Factorial Ratio
    # Cython: expd((lgamma(...) - lgamma(...)) * 0.5)
    # NumPy/SciPy: Use gammaln (Log Gamma function)
    log_numerator = gammaln(p - m - mu + 1)
    log_denominator = gammaln(p + m + mu + 1)
    term_factorial = jnp.exp((log_numerator - log_denominator) * 0.5)

    # Term: Wigner 3j Symbols
    # Note: Passed as m3 = -(m + mu)
    term_w3j_1 = wigner_3j(l, lambda_, p, m, mu, -(m + mu))
    term_w3j_2 = wigner_3j(l, lambda_, q, jnp.zeros_like(q), jnp.zeros_like(q), jnp.zeros_like(q))

    # 5. Combine and Apply Mask
    result = term_normalization * term_phase * term_factorial * term_w3j_1 * term_w3j_2
    
    # Apply the selection rules (set invalid entries to 0)
    # Result is implicitly promoted to complex128 by term_phase
    return jnp.where(zero_mask, 0j, result)


def wigner_3j(j1, j2, j3, m1, m2, m3, max_n=100):
    """
    Computes the Wigner 3-j symbol using JAX.
    
    Parameters:
    -----------
    j1, j2, j3 : int or array_like
        Angular momentum quantum numbers.
    m1, m2, m3 : int or array_like
        Magnetic quantum numbers.
    max_n : int, optional
        Size of the internal summation grid (default 100). 
        Must be larger than j1 + j2 + j3. Increase this for large quantum numbers.
        
    Returns:
    --------
    jax.numpy.ndarray
        The calculated Wigner 3-j coefficients.
    """
    # Ensure inputs are JAX arrays (float for gammaln)
    j1, j2, j3 = map(lambda x: jnp.asarray(x, dtype=jnp.float32), (j1, j2, j3))
    m1, m2, m3 = map(lambda x: jnp.asarray(x, dtype=jnp.float32), (m1, m2, m3))

    # --- 1. Selection Rules (The "Gatekeeper") ---
    # Triangle rule: |j1 - j2| <= j3 <= j1 + j2
    cond_tri = (j3 >= jnp.abs(j1 - j2)) & (j3 <= (j1 + j2))
    
    # Magnetic sum rule: m1 + m2 + m3 == 0
    cond_msum = jnp.isclose(m1 + m2 + m3, 0.0)
    
    # Magnetic magnitude rule: |mi| <= ji
    cond_mlim = (jnp.abs(m1) <= j1) & (jnp.abs(m2) <= j2) & (jnp.abs(m3) <= j3)
    
    # Integer sum rule: j1 + j2 + j3 must be integer
    cond_jsum = jnp.isclose((j1 + j2 + j3) % 1, 0.0)

    # Combined mask (True if valid)
    is_valid = cond_tri & cond_msum & cond_mlim & cond_jsum

    # --- 2. Prefactor Calculation ---
    # Uses log-factorials (gammaln) to prevent overflow
    def log_fact(x):
        return gammaln(x + 1)

    # Triangle coefficient Delta(j1, j2, j3)
    # Delta = (j1+j2-j3)! (j1-j2+j3)! (-j1+j2+j3)! / (j1+j2+j3+1)!
    log_delta = (
        log_fact(j1 + j2 - j3) +
        log_fact(j1 - j2 + j3) +
        log_fact(-j1 + j2 + j3) -
        log_fact(j1 + j2 + j3 + 1)
    )

    # Phase and Root Factors
    # sqrt( (j1+m1)! (j1-m1)! (j2+m2)! (j2-m2)! (j3+m3)! (j3-m3)! )
    log_v = 0.5 * (
        log_fact(j1 + m1) + log_fact(j1 - m1) +
        log_fact(j2 + m2) + log_fact(j2 - m2) +
        log_fact(j3 + m3) + log_fact(j3 - m3)
    )
    
    # Total log prefactor
    log_prefactor = 0.5 * log_delta + log_v
    
    # Sign of the prefactor: (-1)^(j1 - j2 - m3)
    phase_pre = (j1 - j2 - m3)
    sign_pre = jnp.where(phase_pre % 2 != 0, -1.0, 1.0)

    # --- 3. Summation (Racah Formula) ---
    # The sum iterates over integer t.
    # To vectorize in JAX without loops, we compute a fixed grid of t values
    # and mask out the invalid ones (where factorial arguments would be negative).
    
    # Create a grid [0, 1, ..., max_n] for the summation index t
    # We add necessary dimensions for broadcasting if inputs are arrays
    t = jnp.arange(max_n, dtype=jnp.float32)
    
    # Reshape t to (max_n, 1, ...) so it broadcasts against input arrays
    # If inputs are shape (N,), t becomes (max_n, N)
    shape_diff = j1.ndim
    shape_t = (max_n,) + (1,) * shape_diff
    t = t.reshape(shape_t)

    # Factorial arguments in the denominator of the sum:
    # 1: t!
    # 2: (j3 - j2 + t + m1)!
    # 3: (j3 - j1 + t - m2)!
    # 4: (j1 + j2 - j3 - t)!
    # 5: (j1 - t - m1)!
    # 6: (j2 - t + m2)!
    
    arg1 = t
    arg2 = j3 - j2 + t + m1
    arg3 = j3 - j1 + t - m2
    arg4 = j1 + j2 - j3 - t
    arg5 = j1 - t - m1
    arg6 = j2 - t + m2

    # A term is valid only if all factorial arguments are >= 0
    mask_terms = (arg1 >= 0) & (arg2 >= 0) & (arg3 >= 0) & \
                 (arg4 >= 0) & (arg5 >= 0) & (arg6 >= 0)

    # Compute log of the denominator for each t
    # We use a safe mask: if mask is False, use 0.0 to avoid NaN in gammaln, 
    # then zero out the result later.
    log_denom = (
        log_fact(jnp.where(mask_terms, arg1, 0)) +
        log_fact(jnp.where(mask_terms, arg2, 0)) +
        log_fact(jnp.where(mask_terms, arg3, 0)) +
        log_fact(jnp.where(mask_terms, arg4, 0)) +
        log_fact(jnp.where(mask_terms, arg5, 0)) +
        log_fact(jnp.where(mask_terms, arg6, 0))
    )
    
    # Calculate terms in the sum: (-1)^t / denominator
    # magnitude = exp(-log_denom)
    # sign = (-1)^t
    term_mag = jnp.exp(-log_denom)
    term_sign = jnp.where(t % 2 != 0, -1.0, 1.0)
    
    terms = term_sign * term_mag
    
    # Apply mask (set invalid terms to 0)
    terms = jnp.where(mask_terms, terms, 0.0)
    
    # Sum over t (axis 0)
    sum_total = jnp.sum(terms, axis=0)

    # --- 4. Final Combination ---
    result = sign_pre * jnp.exp(log_prefactor) * sum_total
    
    # Apply global selection rules
    return jnp.where(is_valid, result, 0.0)

