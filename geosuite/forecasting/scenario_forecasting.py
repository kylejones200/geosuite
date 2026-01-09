"""
Scenario forecasting with economic inputs.

Provides production forecasting with economic parameters and scenario analysis.
"""
import logging
from typing import Optional, Union, Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScenarioForecaster:
    """
    Scenario forecaster with economic inputs.
    
    Combines production forecasting with economic parameters to generate
    scenario-based forecasts with NPV, revenue, and cost calculations.
    
    Example:
        >>> forecaster = ScenarioForecaster()
        >>> scenarios = forecaster.create_scenarios(
        ...     base_forecast=forecast_df,
        ...     oil_price_scenarios=[60, 80, 100],
        ...     cost_scenarios=[20, 25, 30]
        ... )
    """
    
    def __init__(self):
        """Initialize scenario forecaster."""
        pass
    
    def forecast_with_economics(
        self,
        production_forecast: pd.DataFrame,
        oil_price: float = 60.0,
        gas_price: float = 3.0,
        operating_cost_per_unit: float = 20.0,
        capital_cost: float = 0.0,
        discount_rate: float = 0.10,
        ngl_price: Optional[float] = None,
        production_type: str = 'oil'
    ) -> pd.DataFrame:
        """
        Add economic calculations to production forecast.
        
        Parameters
        ----------
        production_forecast : pd.DataFrame
            Production forecast with 'time' and 'rate' columns
        oil_price : float, default 60.0
            Oil price ($/bbl)
        gas_price : float, default 3.0
            Gas price ($/MCF)
        operating_cost_per_unit : float, default 20.0
            Operating cost per unit of production ($/unit)
        capital_cost : float, default 0.0
            Initial capital cost ($)
        discount_rate : float, default 0.10
            Discount rate for NPV calculation
        ngl_price : float, optional
            NGL price ($/bbl)
        production_type : str, default 'oil'
            Type of production: 'oil', 'gas', or 'mixed'
            
        Returns
        -------
        pd.DataFrame
            Forecast with economic columns:
            - revenue: Revenue per period
            - operating_cost: Operating cost per period
            - net_cash_flow: Net cash flow per period
            - cumulative_npv: Cumulative NPV
        """
        df = production_forecast.copy()
        
        # Calculate revenue
        if production_type == 'oil':
            df['revenue'] = df['rate'] * oil_price
        elif production_type == 'gas':
            df['revenue'] = df['rate'] * gas_price
        else:
            # Mixed - would need separate oil/gas rates
            df['revenue'] = df['rate'] * oil_price  # Simplified
        
        # Calculate operating costs
        df['operating_cost'] = df['rate'] * operating_cost_per_unit
        
        # Net cash flow
        df['net_cash_flow'] = df['revenue'] - df['operating_cost']
        
        # Add initial capital cost to first period
        if capital_cost > 0:
            df.loc[df.index[0], 'net_cash_flow'] -= capital_cost
        
        # Calculate NPV
        df['npv'] = df['net_cash_flow'] / np.power(1 + discount_rate, df['time'] / 365.0)
        df['cumulative_npv'] = df['npv'].cumsum()
        
        # Calculate cumulative revenue and costs
        df['cumulative_revenue'] = df['revenue'].cumsum()
        df['cumulative_cost'] = df['operating_cost'].cumsum()
        if capital_cost > 0:
            df['cumulative_cost'] += capital_cost
        
        return df
    
    def create_scenarios(
        self,
        base_forecast: pd.DataFrame,
        oil_price_scenarios: Optional[List[float]] = None,
        gas_price_scenarios: Optional[List[float]] = None,
        cost_scenarios: Optional[List[float]] = None,
        discount_rate_scenarios: Optional[List[float]] = None,
        production_type: str = 'oil'
    ) -> Dict[str, pd.DataFrame]:
        """
        Create multiple forecast scenarios with different economic inputs.
        
        Parameters
        ----------
        base_forecast : pd.DataFrame
            Base production forecast
        oil_price_scenarios : list of float, optional
            Oil price scenarios ($/bbl)
        gas_price_scenarios : list of float, optional
            Gas price scenarios ($/MCF)
        cost_scenarios : list of float, optional
            Operating cost scenarios ($/unit)
        discount_rate_scenarios : list of float, optional
            Discount rate scenarios
        production_type : str, default 'oil'
            Type of production
            
        Returns
        -------
        dict
            Dictionary of scenario forecasts keyed by scenario name
        """
        scenarios = {}
        
        # Default scenarios if not provided
        if oil_price_scenarios is None:
            oil_price_scenarios = [50, 60, 70, 80]
        if cost_scenarios is None:
            cost_scenarios = [20, 25, 30]
        if discount_rate_scenarios is None:
            discount_rate_scenarios = [0.08, 0.10, 0.12]
        
        scenario_num = 1
        
        for oil_price in oil_price_scenarios:
            for cost in cost_scenarios:
                for dr in discount_rate_scenarios:
                    scenario_name = f"scenario_{scenario_num:03d}_oil${oil_price:.0f}_cost${cost:.0f}_dr{dr:.2f}"
                    
                    forecast = self.forecast_with_economics(
                        base_forecast.copy(),
                        oil_price=oil_price,
                        operating_cost_per_unit=cost,
                        discount_rate=dr,
                        production_type=production_type
                    )
                    
                    scenarios[scenario_name] = forecast
                    scenario_num += 1
        
        logger.info(f"Created {len(scenarios)} forecast scenarios")
        
        return scenarios
    
    def compare_scenarios(
        self,
        scenarios: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios and summarize key metrics.
        
        Parameters
        ----------
        scenarios : dict
            Dictionary of scenario forecasts
            
        Returns
        -------
        pd.DataFrame
            Comparison table with key metrics for each scenario
        """
        comparison = []
        
        for name, forecast in scenarios.items():
            total_npv = forecast['cumulative_npv'].iloc[-1]
            total_revenue = forecast['cumulative_revenue'].iloc[-1]
            total_cost = forecast['cumulative_cost'].iloc[-1]
            total_production = forecast['rate'].sum()
            peak_rate = forecast['rate'].max()
            
            comparison.append({
                'scenario': name,
                'total_npv': total_npv,
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'net_profit': total_revenue - total_cost,
                'total_production': total_production,
                'peak_rate': peak_rate,
                'npv_per_unit': total_npv / total_production if total_production > 0 else 0
            })
        
        return pd.DataFrame(comparison)


def forecast_with_economics(
    production_forecast: pd.DataFrame,
    oil_price: float = 60.0,
    operating_cost_per_unit: float = 20.0,
    discount_rate: float = 0.10,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to add economics to production forecast.
    
    Parameters
    ----------
    production_forecast : pd.DataFrame
        Production forecast
    oil_price : float, default 60.0
        Oil price ($/bbl)
    operating_cost_per_unit : float, default 20.0
        Operating cost per unit
    discount_rate : float, default 0.10
        Discount rate
    **kwargs
        Additional economic parameters
        
    Returns
    -------
    pd.DataFrame
        Forecast with economics
    """
    forecaster = ScenarioForecaster()
    return forecaster.forecast_with_economics(
        production_forecast,
        oil_price=oil_price,
        operating_cost_per_unit=operating_cost_per_unit,
        discount_rate=discount_rate,
        **kwargs
    )


def create_scenarios(
    base_forecast: pd.DataFrame,
    oil_price_scenarios: Optional[List[float]] = None,
    cost_scenarios: Optional[List[float]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to create forecast scenarios.
    
    Parameters
    ----------
    base_forecast : pd.DataFrame
        Base production forecast
    oil_price_scenarios : list of float, optional
        Oil price scenarios
    cost_scenarios : list of float, optional
        Cost scenarios
    **kwargs
        Additional parameters
        
    Returns
    -------
    dict
        Scenario forecasts
    """
    forecaster = ScenarioForecaster()
    return forecaster.create_scenarios(
        base_forecast,
        oil_price_scenarios=oil_price_scenarios,
        cost_scenarios=cost_scenarios,
        **kwargs
    )


