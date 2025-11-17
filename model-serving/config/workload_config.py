"""
Workload Generation Configuration.

Contains parameters for emotion-aware task generation, service time mapping,
arrival rate modification, and emotion configuration.
"""
import os


# ============================================================================
# SERVICE TIME MAPPING PARAMETERS
# ============================================================================

# Service Time Mapping Parameters
# Formula: S_i = L_0 * (1 + ALPHA * arousal_i)
BASE_SERVICE_TIME = 1.0  # L_0: Base service time when arousal = 0
ALPHA = 0.5              # α: Sensitivity coefficient for arousal impact on service time
GAMMA_NONLINEAR = 1.0    # γ: Non-linear exponent (1.0 = linear mapping)
RHO = 1.0                # ρ: Correlation strength between emotion and service time
MIN_SERVICE_TIME = 0.1   # Minimum allowed service time (safety bound)

# Environment variable overrides
BASE_SERVICE_TIME = float(os.environ.get('BASE_SERVICE_TIME_ENV', BASE_SERVICE_TIME))
ALPHA = float(os.environ.get('ALPHA_ENV', ALPHA))
GAMMA_NONLINEAR = float(os.environ.get('GAMMA_NONLINEAR_ENV', GAMMA_NONLINEAR))
RHO = float(os.environ.get('RHO_ENV', RHO))


# ============================================================================
# ARRIVAL RATE MODIFICATION PARAMETERS
# ============================================================================

# Arrival Rate Modification Parameters
# Formula: λ(a) = λ_0 * (1 + GAMMA * arousal)
BASE_ARRIVAL_RATE = 2.0  # λ_0: Base arrival rate when arousal = 0
GAMMA = 0.3              # γ: Sensitivity coefficient for arousal impact on arrival rate

# Environment variable overrides
BASE_ARRIVAL_RATE = float(os.environ.get('BASE_ARRIVAL_RATE_ENV', BASE_ARRIVAL_RATE))
GAMMA = float(os.environ.get('GAMMA_ENV', GAMMA))


# ============================================================================
# EMOTION CONFIGURATION
# ============================================================================

# Emotion Configuration
AROUSAL_NOISE_STD = 0.0  # Standard deviation of noise added to arousal values
SERVICE_TIME_MAPPING_FUNC = 'linear'  # 'linear', 'exponential', or 'gamma_dist'

# Environment variable overrides
AROUSAL_NOISE_STD = float(os.environ.get('AROUSAL_NOISE_STD_ENV', AROUSAL_NOISE_STD))
SERVICE_TIME_MAPPING_FUNC = os.environ.get('SERVICE_TIME_MAPPING_ENV', SERVICE_TIME_MAPPING_FUNC)


# ============================================================================
# WORKLOAD PARAMETER DOCUMENTATION
# ============================================================================
"""
Parameter Documentation for Emotion-aware Workload Generation:

1. BASE_SERVICE_TIME (L_0):
   - Range: > 0
   - Default: 2.0
   - Description: Baseline service time when arousal is neutral (0)

2. ALPHA (α):
   - Range: 0 < α ≤ 1
   - Default: 0.5
   - Description: Controls how much arousal affects service time
   - Higher α = stronger arousal impact

3. GAMMA (γ):
   - Range: 0 ≤ γ < 1
   - Default: 0.3
   - Description: Controls how much arousal affects arrival rate
   - Higher γ = higher arrival rate for high-arousal emotions

4. RHO (ρ):
   - Range: 0 ≤ ρ ≤ 1
   - Default: 1.0
   - Description: Correlation strength between emotion and system behavior
   - ρ = 0: no emotion effect, ρ = 1: full emotion effect

5. SERVICE_TIME_MAPPING_FUNC:
   - Options: 'linear', 'exponential', 'gamma_dist'
   - Default: 'linear'
   - Description: Function to map arousal to service time

Example Usage:
   # Run with custom parameters
   export ALPHA_ENV=0.7
   export GAMMA_ENV=0.5
   export BASE_SERVICE_TIME_ENV=3.0
   python run_simulation.py --num_jobs 100
"""
