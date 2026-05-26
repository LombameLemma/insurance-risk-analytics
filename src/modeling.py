"""ML modeling utilities for insurance analytics."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceRiskModel:
    """Risk prediction models for insurance data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize modeling pipeline.

        Args:
            df: Insurance data DataFrame
        """
        self.df = df
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def prepare_data(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare and split data for modeling.

        Args:
            target_col: Target variable column
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Create target variable
        y = (self.df[target_col] > 0).astype(int)
        
        # Select features
        X = self.df.select_dtypes(include=[np.number]).copy()
        
        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: pd.Series
    ) -> Dict:
        """
        Train logistic regression model.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Model performance metrics
        """
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        self.models["logistic_regression"] = model
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Logistic Regression - Train Accuracy: {train_score:.4f}")
        
        return {"model": model, "train_accuracy": train_score}

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        n_estimators: int = 100
    ) -> Dict:
        """
        Train random forest model.

        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees

        Returns:
            Model performance metrics
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models["random_forest"] = model
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Random Forest - Train Accuracy: {train_score:.4f}")
        
        return {"model": model, "train_accuracy": train_score}

    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate all trained models.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with evaluation metrics for all models
        """
        results = {}
        
        for model_name, model in self.models.items():
            accuracy = model.score(X_test, y_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                "accuracy": accuracy,
                "auc_score": auc_score
            }
            
            logger.info(
                f"{model_name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}"
            )
        
        return results
