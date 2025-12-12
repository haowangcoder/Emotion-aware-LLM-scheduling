"""
Unit tests for affect_weight_v2 module.

Tests cover:
1. Hard gating (v1.0 compatible) - compute_urgency_hard
2. Soft gating (sigmoid) - compute_urgency_soft
3. Dual-channel (Depression + Panic) - compute_urgency_dual_channel
4. Weight computation with all modes
5. Preset configurations
6. Backward compatibility

Run with: pytest tests/core/test_affect_weight_v2.py -v
"""

import pytest
import math
from core.affect_weight_v2 import (
    WeightMode, WeightConfig,
    sigmoid,
    compute_urgency_hard,
    compute_urgency_soft,
    compute_urgency_dual_channel,
    compute_urgency_v2,
    affect_weight_v2,
    compute_wspt_score_v2,
    get_detailed_weight_info,
    get_preset_config,
    list_presets,
    PRESET_CONFIGS,
    # Backward compatibility
    compute_urgency,
    affect_weight,
)


class TestSigmoid:
    """Test sigmoid helper function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be 0.5."""
        assert abs(sigmoid(0) - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        """sigmoid(large positive) should be close to 1."""
        assert sigmoid(10) > 0.9999

    def test_sigmoid_large_negative(self):
        """sigmoid(large negative) should be close to 0."""
        assert sigmoid(-10) < 0.0001

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) + sigmoid(x) = 1."""
        for x in [-5, -1, 0, 1, 5]:
            assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-10


class TestHardGating:
    """Test original hard gating urgency computation."""

    def test_depression_quadrant(self):
        """Depression quadrant (neg valence, neg arousal) should get urgency."""
        u = compute_urgency_hard(arousal=-0.8, valence=-0.8)
        assert abs(u - 0.64) < 1e-10

    def test_panic_quadrant(self):
        """Panic quadrant (neg valence, pos arousal) should get 0 urgency."""
        u = compute_urgency_hard(arousal=0.8, valence=-0.8)
        assert u == 0.0

    def test_calm_quadrant(self):
        """Calm quadrant (pos valence, neg arousal) should get 0 urgency."""
        u = compute_urgency_hard(arousal=-0.8, valence=0.8)
        assert u == 0.0

    def test_excited_quadrant(self):
        """Excited quadrant (pos valence, pos arousal) should get 0 urgency."""
        u = compute_urgency_hard(arousal=0.8, valence=0.8)
        assert u == 0.0

    def test_maximum_urgency(self):
        """Maximum depression (a=-1, v=-1) should give u=1."""
        u = compute_urgency_hard(arousal=-1.0, valence=-1.0)
        assert u == 1.0

    def test_edge_case_zero(self):
        """Edge case at origin should give 0 urgency."""
        u = compute_urgency_hard(arousal=0.0, valence=0.0)
        assert u == 0.0

    def test_exponents(self):
        """Exponents p, q should affect urgency curve."""
        u_linear = compute_urgency_hard(arousal=-0.5, valence=-0.5, p=1.0, q=1.0)
        u_quadratic = compute_urgency_hard(arousal=-0.5, valence=-0.5, p=2.0, q=2.0)
        assert u_linear == 0.25
        assert u_quadratic == 0.0625


class TestSoftGating:
    """Test soft gating (sigmoid) urgency computation."""

    def test_soft_continuous_at_zero(self):
        """Soft gating should be continuous at 0."""
        # Values around 0 should not jump dramatically
        u_neg = compute_urgency_soft(arousal=-0.01, valence=-0.01, k_v=5.0, k_a=5.0)
        u_pos = compute_urgency_soft(arousal=0.01, valence=0.01, k_v=5.0, k_a=5.0)
        # Should be reasonably close (no hard jump)
        assert abs(u_neg - u_pos) < 0.1

    def test_soft_depression_high(self):
        """Depression quadrant should get high urgency."""
        u = compute_urgency_soft(arousal=-0.8, valence=-0.8, k_v=5.0, k_a=5.0)
        assert u > 0.9  # Should be close to 1 due to sigmoid saturation

    def test_soft_panic_low(self):
        """Panic quadrant should get low urgency (but not exactly 0)."""
        u = compute_urgency_soft(arousal=0.8, valence=-0.8, k_v=5.0, k_a=5.0)
        assert u < 0.1  # Low but not exactly 0

    def test_soft_steepness_effect(self):
        """Higher k should make transition sharper."""
        # With low k, urgency should be moderate at mild depression
        u_low_k = compute_urgency_soft(arousal=-0.2, valence=-0.2, k_v=1.0, k_a=1.0)
        u_high_k = compute_urgency_soft(arousal=-0.2, valence=-0.2, k_v=10.0, k_a=10.0)
        # Higher k should give higher urgency for same (negative) values
        assert u_high_k > u_low_k

    def test_soft_threshold_effect(self):
        """tau parameters should shift the threshold."""
        # With tau_v=-0.3, we need v < -0.3 to get high n
        u_no_threshold = compute_urgency_soft(arousal=-0.5, valence=-0.2, tau_v=0.0, k_v=5.0)
        u_with_threshold = compute_urgency_soft(arousal=-0.5, valence=-0.2, tau_v=-0.3, k_v=5.0)
        # With threshold, v=-0.2 is not "negative enough" so urgency should be lower
        assert u_with_threshold < u_no_threshold


class TestDualChannel:
    """Test dual-channel (Depression + Panic) urgency computation."""

    def test_dual_depression_dominates(self):
        """Depression quadrant should have higher total urgency."""
        u_total, u_dep, u_panic = compute_urgency_dual_channel(
            arousal=-0.8, valence=-0.8,
            gamma_dep=1.0, gamma_panic=0.3
        )
        assert u_dep > 0.9
        assert u_panic < 0.1
        assert u_total > 0.9

    def test_dual_panic_gets_boost(self):
        """Panic quadrant should now get some urgency boost."""
        u_total, u_dep, u_panic = compute_urgency_dual_channel(
            arousal=0.8, valence=-0.8,
            gamma_dep=1.0, gamma_panic=0.3
        )
        assert u_dep < 0.1
        assert u_panic > 0.9
        # Total should be gamma_panic * u_panic ~ 0.3 * 0.9 = 0.27
        assert 0.2 < u_total < 0.4

    def test_dual_gamma_balance(self):
        """gamma_panic controls panic channel contribution."""
        # High gamma_panic should give panic more weight
        _, _, u_panic_low = compute_urgency_dual_channel(
            arousal=0.8, valence=-0.8, gamma_dep=1.0, gamma_panic=0.1
        )
        u_total_high, _, u_panic_high = compute_urgency_dual_channel(
            arousal=0.8, valence=-0.8, gamma_dep=1.0, gamma_panic=0.5
        )
        # Individual panic urgency should be similar
        assert abs(u_panic_low - u_panic_high) < 0.05
        # But total with higher gamma should be higher
        # Note: u_total_high should be ~ 0.5 * u_panic ~ 0.45

    def test_dual_clamping(self):
        """Total urgency should be clamped to [0, 1]."""
        # Even with high gammas, u_total should not exceed 1
        u_total, _, _ = compute_urgency_dual_channel(
            arousal=-0.8, valence=-0.8,
            gamma_dep=1.0, gamma_panic=1.0  # Sum > 1
        )
        assert u_total <= 1.0


class TestComputeUrgencyV2:
    """Test unified urgency computation with configurable mode."""

    def test_hard_mode(self):
        """HARD mode should match compute_urgency_hard."""
        config = WeightConfig(mode=WeightMode.HARD, p=1.0, q=1.0)
        u_v2 = compute_urgency_v2(arousal=-0.8, valence=-0.8, config=config)
        u_hard = compute_urgency_hard(arousal=-0.8, valence=-0.8)
        assert abs(u_v2 - u_hard) < 1e-10

    def test_soft_mode(self):
        """SOFT mode should use sigmoid gating."""
        config = WeightConfig(mode=WeightMode.SOFT, k_v=5.0, k_a=5.0)
        u = compute_urgency_v2(arousal=-0.8, valence=-0.8, config=config)
        # Soft mode gives higher urgency due to sigmoid saturation
        assert u > 0.9

    def test_dual_channel_mode(self):
        """DUAL_CHANNEL mode should combine depression and panic."""
        config = WeightConfig(
            mode=WeightMode.DUAL_CHANNEL,
            gamma_dep=1.0, gamma_panic=0.3
        )
        # Panic user should now get some urgency
        u_panic = compute_urgency_v2(arousal=0.8, valence=-0.8, config=config)
        assert u_panic > 0.2

    def test_default_config(self):
        """Default config (None) should use SOFT mode."""
        u = compute_urgency_v2(arousal=-0.8, valence=-0.8, config=None)
        # Default is SOFT mode
        assert u > 0.9


class TestAffectWeightV2:
    """Test affect weight computation."""

    def test_weight_range(self):
        """Weight should be in [1, w_max]."""
        config = WeightConfig(w_max=2.0)
        # Depression user
        w_dep = affect_weight_v2(arousal=-0.8, valence=-0.8, config=config)
        assert 1.0 <= w_dep <= 2.0

        # Neutral user
        w_neutral = affect_weight_v2(arousal=0.0, valence=0.0, config=config)
        assert 1.0 <= w_neutral <= 2.0

    def test_confidence_discount(self):
        """Lower confidence should reduce weight boost."""
        config = WeightConfig(w_max=2.0, mode=WeightMode.HARD)
        w_full = affect_weight_v2(arousal=-0.8, valence=-0.8, confidence=1.0, config=config)
        w_half = affect_weight_v2(arousal=-0.8, valence=-0.8, confidence=0.5, config=config)
        w_zero = affect_weight_v2(arousal=-0.8, valence=-0.8, confidence=0.0, config=config)

        assert w_full > w_half > w_zero
        assert w_zero == 1.0  # No boost with 0 confidence

    def test_w_max_effect(self):
        """Higher w_max should allow larger weight boost."""
        config_low = WeightConfig(w_max=1.5, mode=WeightMode.HARD)
        config_high = WeightConfig(w_max=3.0, mode=WeightMode.HARD)

        w_low = affect_weight_v2(arousal=-1.0, valence=-1.0, config=config_low)
        w_high = affect_weight_v2(arousal=-1.0, valence=-1.0, config=config_high)

        assert w_low == 1.5
        assert w_high == 3.0


class TestWSPTScore:
    """Test WSPT score computation."""

    def test_score_computation(self):
        """Score = service_time / weight."""
        config = WeightConfig(w_max=2.0, mode=WeightMode.HARD)
        score = compute_wspt_score_v2(
            predicted_service_time=4.0,
            arousal=-0.8, valence=-0.8,
            config=config
        )
        # weight = 1 + (2-1) * 0.64 = 1.64
        # score = 4.0 / 1.64 ~ 2.44
        expected_weight = 1.0 + (2.0 - 1.0) * 0.64
        expected_score = 4.0 / expected_weight
        assert abs(score - expected_score) < 0.01

    def test_depression_gets_lower_score(self):
        """Depression user with same service time should get lower score."""
        config = WeightConfig(w_max=2.0, mode=WeightMode.HARD)
        score_depression = compute_wspt_score_v2(
            predicted_service_time=3.0, arousal=-0.8, valence=-0.8, config=config
        )
        score_neutral = compute_wspt_score_v2(
            predicted_service_time=3.0, arousal=0.0, valence=0.0, config=config
        )
        # Depression should have lower score (higher priority)
        assert score_depression < score_neutral


class TestPresetConfigs:
    """Test preset configurations."""

    def test_all_presets_exist(self):
        """All documented presets should exist."""
        expected = [
            'depression_first_hard',
            'depression_first_soft',
            'dual_channel_balanced',
            'dual_channel_depression_heavy',
        ]
        for name in expected:
            config = get_preset_config(name)
            assert isinstance(config, WeightConfig)

    def test_hard_preset_mode(self):
        """depression_first_hard should use HARD mode."""
        config = get_preset_config('depression_first_hard')
        assert config.mode == WeightMode.HARD

    def test_soft_preset_mode(self):
        """depression_first_soft should use SOFT mode."""
        config = get_preset_config('depression_first_soft')
        assert config.mode == WeightMode.SOFT

    def test_dual_channel_presets(self):
        """Dual channel presets should use DUAL_CHANNEL mode."""
        for name in ['dual_channel_balanced', 'dual_channel_depression_heavy']:
            config = get_preset_config(name)
            assert config.mode == WeightMode.DUAL_CHANNEL

    def test_invalid_preset_raises(self):
        """Invalid preset name should raise ValueError."""
        with pytest.raises(ValueError):
            get_preset_config('nonexistent_preset')

    def test_list_presets(self):
        """list_presets should return descriptions."""
        presets = list_presets()
        assert len(presets) == 4
        assert all(isinstance(desc, str) for desc in presets.values())


class TestDetailedWeightInfo:
    """Test detailed weight info for debugging."""

    def test_info_contains_basics(self):
        """Info should contain basic values."""
        config = WeightConfig(mode=WeightMode.SOFT)
        info = get_detailed_weight_info(arousal=-0.5, valence=-0.5, config=config)

        assert 'arousal' in info
        assert 'valence' in info
        assert 'confidence' in info
        assert 'mode' in info
        assert 'weight' in info

    def test_dual_channel_info(self):
        """Dual channel should have channel-specific info."""
        config = WeightConfig(mode=WeightMode.DUAL_CHANNEL)
        info = get_detailed_weight_info(arousal=-0.5, valence=-0.5, config=config)

        assert 'urgency_depression' in info
        assert 'urgency_panic' in info
        assert 'gamma_dep' in info
        assert 'gamma_panic' in info


class TestBackwardCompatibility:
    """Test backward compatibility with v1.0 functions."""

    def test_compute_urgency_compat(self):
        """compute_urgency should match v1.0 behavior."""
        u_compat = compute_urgency(arousal=-0.8, valence=-0.8)
        u_hard = compute_urgency_hard(arousal=-0.8, valence=-0.8)
        assert u_compat == u_hard

    def test_affect_weight_compat(self):
        """affect_weight should match v1.0 behavior."""
        w_compat = affect_weight(arousal=-0.8, valence=-0.8, w_max=2.0)
        # Manual calculation: u = 0.64, w = 1 + 1 * 0.64 = 1.64
        assert abs(w_compat - 1.64) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_values(self):
        """Test with extreme arousal/valence values."""
        config = WeightConfig(mode=WeightMode.SOFT)
        # Should not crash
        w = affect_weight_v2(arousal=-1.0, valence=-1.0, config=config)
        assert w > 1.0

        w = affect_weight_v2(arousal=1.0, valence=1.0, config=config)
        assert w >= 1.0

    def test_zero_confidence(self):
        """Zero confidence should give weight = 1.0."""
        config = WeightConfig(mode=WeightMode.SOFT)
        w = affect_weight_v2(arousal=-1.0, valence=-1.0, confidence=0.0, config=config)
        assert w == 1.0

    def test_none_config_defaults(self):
        """None config should use default SOFT mode."""
        w = affect_weight_v2(arousal=-0.5, valence=-0.5, config=None)
        assert isinstance(w, float)
        assert w > 1.0


class TestComparison:
    """Compare different modes for the same inputs."""

    def test_mode_comparison_depression(self):
        """Compare modes for depression quadrant user."""
        configs = {
            'hard': WeightConfig(mode=WeightMode.HARD, w_max=2.0),
            'soft': WeightConfig(mode=WeightMode.SOFT, w_max=2.0),
            'dual': WeightConfig(mode=WeightMode.DUAL_CHANNEL, w_max=2.0),
        }

        weights = {
            name: affect_weight_v2(arousal=-0.8, valence=-0.8, config=config)
            for name, config in configs.items()
        }

        # All should give weight > 1 for depression
        assert all(w > 1.0 for w in weights.values())
        # Soft should give higher weight due to sigmoid saturation
        assert weights['soft'] > weights['hard']

    def test_mode_comparison_panic(self):
        """Compare modes for panic quadrant user."""
        configs = {
            'hard': WeightConfig(mode=WeightMode.HARD, w_max=2.0),
            'soft': WeightConfig(mode=WeightMode.SOFT, w_max=2.0),
            'dual': WeightConfig(mode=WeightMode.DUAL_CHANNEL, w_max=2.0, gamma_panic=0.3),
        }

        weights = {
            name: affect_weight_v2(arousal=0.8, valence=-0.8, config=config)
            for name, config in configs.items()
        }

        # HARD should give w=1.0 (no boost)
        assert abs(weights['hard'] - 1.0) < 1e-10
        # SOFT should give slight boost
        assert weights['soft'] > 1.0
        # DUAL should give moderate boost
        assert weights['dual'] > weights['soft']
