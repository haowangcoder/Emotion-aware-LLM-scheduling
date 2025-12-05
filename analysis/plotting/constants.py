"""
Constants for publication-quality plotting.

Includes Okabe-Ito colorblind-friendly color scheme and consistent styling.
"""

# =============================================================================
# Okabe-Ito Color Scheme (Colorblind-Friendly)
# =============================================================================

COLORS = {
    'FCFS': '#4D4D4D',           # Dark gray
    'SSJF-Emotion': '#E69F00',   # Orange
    'SSJF-Combined': '#0072B2',  # Blue
    'SSJF-Valence': '#009E73',   # Green
}

MARKERS = {
    'FCFS': 'o',           # Circle
    'SSJF-Emotion': 's',   # Square
    'SSJF-Combined': '^',  # Triangle
    'SSJF-Valence': 'D',   # Diamond
}

# Consistent ordering across all plots
SCHEDULER_ORDER = ['FCFS', 'SSJF-Emotion', 'SSJF-Combined', 'SSJF-Valence']

# Arousal class ordering and labels
AROUSAL_ORDER = ['low', 'medium', 'high']
AROUSAL_LABELS = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}

# Valence class ordering and labels
VALENCE_ORDER = ['negative', 'neutral', 'positive']
VALENCE_LABELS = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}

# Arousal colors for calibration plots
AROUSAL_COLORS = {
    'low': '#6bcf7f',     # Green
    'medium': '#ffd93d',  # Yellow
    'high': '#ff6b6b',    # Red
}


def get_scheduler_color(name: str) -> str:
    """Get color for a scheduler, with fallback."""
    return COLORS.get(name, '#888888')


def get_scheduler_marker(name: str) -> str:
    """Get marker for a scheduler, with fallback."""
    return MARKERS.get(name, 'o')
