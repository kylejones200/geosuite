"""
Forecasting and decline curve analysis module.

Provides tools for production forecasting including:
- Physics-informed decline models
- Bayesian posterior sampling
- Time series decomposition
- Scenario forecasting with economic inputs
- Monte Carlo ensembles
"""

__all__ = []

try:
    from .decline_models import (
        DeclineModel,
        ExponentialDecline,
        HyperbolicDecline,
        HarmonicDecline,
        fit_decline_model,
        forecast_production
    )
    __all__.extend([
        'DeclineModel',
        'ExponentialDecline',
        'HyperbolicDecline',
        'HarmonicDecline',
        'fit_decline_model',
        'forecast_production'
    ])
except ImportError:
    pass

try:
    from .bayesian_decline import (
        BayesianDeclineAnalyzer,
        sample_decline_posterior,
        forecast_with_uncertainty
    )
    __all__.extend([
        'BayesianDeclineAnalyzer',
        'sample_decline_posterior',
        'forecast_with_uncertainty'
    ])
except ImportError:
    pass

try:
    from .decomposition import (
        decompose_production,
        detect_trend,
        detect_seasonality,
        remove_trend_seasonality
    )
    __all__.extend([
        'decompose_production',
        'detect_trend',
        'detect_seasonality',
        'remove_trend_seasonality'
    ])
except ImportError:
    pass

try:
    from .scenario_forecasting import (
        ScenarioForecaster,
        forecast_with_economics,
        create_scenarios
    )
    __all__.extend([
        'ScenarioForecaster',
        'forecast_with_economics',
        'create_scenarios'
    ])
except ImportError:
    pass

try:
    from .monte_carlo_forecast import (
        MonteCarloForecaster,
        ensemble_forecast,
        forecast_uncertainty_bands
    )
    __all__.extend([
        'MonteCarloForecaster',
        'ensemble_forecast',
        'forecast_uncertainty_bands'
    ])
except ImportError:
    pass

