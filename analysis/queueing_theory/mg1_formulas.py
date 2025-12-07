"""
M/G/1 Queueing Theory Formulas

Implements the Pollaczek-Khinchin (P-K) formula and related calculations
for M/G/1 queues.

Reference:
    Kleinrock, L. (1975). Queueing Systems, Volume I: Theory.
"""

import numpy as np
from typing import Dict, Optional


def pollaczek_khinchin(
    lambda_arrival: float,
    E_S: float,
    E_S2: float
) -> Dict[str, float]:
    """
    Compute M/G/1 performance metrics using Pollaczek-Khinchin formula.

    The P-K formula gives the mean waiting time in queue:
        W_q = λ·E[S²] / [2(1-ρ)]

    Args:
        lambda_arrival: Arrival rate λ (jobs/s)
        E_S: Mean service time E[S] (seconds)
        E_S2: Second moment of service time E[S²] (seconds²)

    Returns:
        dict with keys:
            - rho: System utilization ρ = λ·E[S]
            - W_q: Mean waiting time in queue (seconds)
            - W: Mean system time / response time (seconds)
            - L_q: Mean queue length (jobs)
            - L: Mean system size (jobs)

    Example:
        >>> result = pollaczek_khinchin(lambda_arrival=0.5, E_S=1.5, E_S2=2.89)
        >>> print(f"Utilization: {result['rho']:.2f}")
        >>> print(f"Mean waiting time: {result['W_q']:.2f}s")
    """
    rho = lambda_arrival * E_S

    if rho >= 1.0:
        return {
            'rho': rho,
            'W_q': float('inf'),
            'W': float('inf'),
            'L_q': float('inf'),
            'L': float('inf'),
            'stable': False
        }

    # Pollaczek-Khinchin formula
    W_q = (lambda_arrival * E_S2) / (2 * (1 - rho))

    # Mean system time (response time)
    W = W_q + E_S

    # Little's Law: L = λW
    L_q = lambda_arrival * W_q
    L = lambda_arrival * W

    return {
        'rho': rho,
        'W_q': W_q,
        'W': W,
        'L_q': L_q,
        'L': L,
        'stable': True
    }


def compute_service_time_moments(
    service_times: np.ndarray,
    ddof: int = 1
) -> Dict[str, float]:
    """
    Compute service time distribution moments from empirical data.

    Args:
        service_times: Array of observed service times
        ddof: Delta degrees of freedom for variance calculation
              (default=1 for sample variance)

    Returns:
        dict with:
            - E_S: Mean service time
            - E_S2: Second moment E[S²]
            - Var_S: Variance of service time
            - std_S: Standard deviation
            - C_s: Coefficient of variation (σ/μ)
            - min: Minimum service time
            - max: Maximum service time
            - n: Sample size

    Example:
        >>> times = np.array([1.0, 1.5, 2.0, 1.2, 1.8])
        >>> moments = compute_service_time_moments(times)
        >>> print(f"Mean: {moments['E_S']:.3f}, CV: {moments['C_s']:.3f}")
    """
    service_times = np.asarray(service_times)

    E_S = np.mean(service_times)
    E_S2 = np.mean(service_times ** 2)
    Var_S = np.var(service_times, ddof=ddof)
    std_S = np.sqrt(Var_S)

    # Coefficient of variation
    C_s = std_S / E_S if E_S > 0 else 0.0

    return {
        'E_S': float(E_S),
        'E_S2': float(E_S2),
        'Var_S': float(Var_S),
        'std_S': float(std_S),
        'C_s': float(C_s),
        'min': float(np.min(service_times)),
        'max': float(np.max(service_times)),
        'n': len(service_times)
    }


def theoretical_service_time_moments(
    base_service_time: float,
    alpha: float,
    arousal_values: Dict[str, float],
    class_proportions: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute theoretical service time moments from model parameters.

    Service time formula: S_c = L_0 × (1 + α × a_c)

    Args:
        base_service_time: L_0 (base service time in seconds)
        alpha: α coefficient for arousal-service time mapping
        arousal_values: {class_name: arousal_value}
                       e.g., {'low': -0.8, 'medium': 0.0, 'high': 0.8}
        class_proportions: {class_name: proportion}
                          Defaults to uniform distribution

    Returns:
        dict with:
            - per_class: {class_name: {E_S, E_S2}}
            - overall: {E_S, E_S2, Var_S, C_s}
    """
    if class_proportions is None:
        n_classes = len(arousal_values)
        class_proportions = {c: 1.0 / n_classes for c in arousal_values}

    # Compute per-class service times (deterministic given arousal)
    per_class = {}
    for class_name, arousal in arousal_values.items():
        S_c = base_service_time * (1 + alpha * arousal)
        per_class[class_name] = {
            'arousal': arousal,
            'E_S': S_c,
            'E_S2': S_c ** 2,  # Deterministic, so E[S²] = S²
            'proportion': class_proportions[class_name]
        }

    # Compute overall moments
    E_S = sum(
        per_class[c]['proportion'] * per_class[c]['E_S']
        for c in arousal_values
    )

    E_S2 = sum(
        per_class[c]['proportion'] * per_class[c]['E_S2']
        for c in arousal_values
    )

    Var_S = E_S2 - E_S ** 2
    std_S = np.sqrt(max(0, Var_S))
    C_s = std_S / E_S if E_S > 0 else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'E_S': E_S,
            'E_S2': E_S2,
            'Var_S': Var_S,
            'std_S': std_S,
            'C_s': C_s
        }
    }


def pk_sensitivity_analysis(
    lambda_arrival: float,
    E_S: float,
    E_S2: float,
    delta_rho: float = 0.01
) -> Dict[str, float]:
    """
    Analyze sensitivity of P-K formula to utilization changes.

    At high loads (ρ → 1), small changes in ρ cause large changes in W_q.
    This function quantifies this sensitivity.

    Args:
        lambda_arrival: Arrival rate
        E_S: Mean service time
        E_S2: Second moment
        delta_rho: Small change in ρ for numerical differentiation

    Returns:
        dict with:
            - rho: Current utilization
            - dW_q_drho: Derivative of W_q with respect to ρ
            - sensitivity_factor: 1/(1-ρ)² (theoretical sensitivity)
            - pct_change_per_pct_rho: % change in W_q per 1% change in ρ
    """
    rho = lambda_arrival * E_S

    if rho >= 1.0:
        return {
            'rho': rho,
            'dW_q_drho': float('inf'),
            'sensitivity_factor': float('inf'),
            'pct_change_per_pct_rho': float('inf')
        }

    # Current W_q
    W_q = (lambda_arrival * E_S2) / (2 * (1 - rho))

    # Theoretical sensitivity: dW_q/dρ ∝ 1/(1-ρ)²
    sensitivity_factor = 1.0 / ((1 - rho) ** 2)

    # Numerical derivative
    rho_plus = min(rho + delta_rho, 0.999)
    lambda_plus = rho_plus / E_S
    W_q_plus = (lambda_plus * E_S2) / (2 * (1 - rho_plus))

    dW_q_drho = (W_q_plus - W_q) / delta_rho

    # Percentage change per 1% change in rho
    pct_change = (dW_q_drho * 0.01 / W_q) * 100 if W_q > 0 else float('inf')

    return {
        'rho': rho,
        'W_q': W_q,
        'dW_q_drho': dW_q_drho,
        'sensitivity_factor': sensitivity_factor,
        'pct_change_per_pct_rho': pct_change
    }
