"""Hypothesis testing utilities for insurance analytics."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypothesisTests:
    """Statistical hypothesis tests for insurance data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize hypothesis tests with data.

        Args:
            df: Insurance data DataFrame
        """
        self.df = df
        self.results = {}

    def test_loss_ratio_by_category(
        self, category_col: str, significance_level: float = 0.05
    ) -> Dict:
        """
        Test if loss ratios differ significantly across categories.

        Args:
            category_col: Categorical column to test
            significance_level: Alpha level for significance

        Returns:
            Dictionary with test results
        """
        if "LossRatio" not in self.df.columns:
            self.df["LossRatio"] = (
                self.df["TotalClaims"] / self.df["TotalPremium"]
            )

        groups = [group["LossRatio"].values 
                  for name, group in self.df.groupby(category_col)]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        result = {
            "test": "ANOVA",
            "category": category_col,
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < significance_level,
            "alpha": significance_level,
        }
        
        logger.info(f"ANOVA test for {category_col}: p-value={p_value:.6f}")
        return result

    def test_claim_frequency_difference(
        self, group_col: str, significance_level: float = 0.05
    ) -> Dict:
        """
        Test if claim frequencies differ between groups.

        Args:
            group_col: Grouping column
            significance_level: Alpha level

        Returns:
            Dictionary with test results
        """
        groups = self.df.groupby(group_col)
        
        contingency_data = []
        for name, group in groups:
            has_claims = (group["TotalClaims"] > 0).sum()
            no_claims = (group["TotalClaims"] == 0).sum()
            contingency_data.append([has_claims, no_claims])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_data)
        
        result = {
            "test": "Chi-Square",
            "group": group_col,
            "chi2_statistic": chi2,
            "p_value": p_value,
            "significant": p_value < significance_level,
            "alpha": significance_level,
        }
        
        logger.info(f"Chi-Square test for {group_col}: p-value={p_value:.6f}")
        return result
