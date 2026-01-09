"""
Deep learning models for subsurface data analysis.

Provides neural network implementations for facies classification and
petrophysical property prediction with built-in explainability support.
"""
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

from ..base.estimators import BaseEstimator

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. Deep models require PyTorch. "
        "Install with: pip install torch"
    )
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None

# Try to import TensorFlow as fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class WellLogDataset:
    """Dataset wrapper for well log data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        scaler: Optional[StandardScaler] = None
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Feature array
        y : np.ndarray, optional
            Target array
        scaler : StandardScaler, optional
            Pre-fitted scaler for features
        """
        self.X = torch.FloatTensor(X) if TORCH_AVAILABLE else X
        self.y = torch.FloatTensor(y) if y is not None and TORCH_AVAILABLE else y
        self.scaler = scaler
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class DeepFaciesClassifier(BaseEstimator):
    """
    Deep neural network for facies classification.
    
    Multi-layer perceptron with configurable architecture for classifying
    facies from well log data. Supports dropout, batch normalization, and
    early stopping.
    
    Example:
        >>> from geosuite.ml import DeepFaciesClassifier
        >>> 
        >>> model = DeepFaciesClassifier(
        ...     hidden_layers=[128, 64, 32],
        ...     dropout=0.3,
        ...     learning_rate=0.001
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        random_state: int = 42,
        use_torch: bool = True
    ):
        """
        Initialize deep facies classifier.
        
        Parameters
        ----------
        hidden_layers : list of int, default [128, 64, 32]
            Number of neurons in each hidden layer
        dropout : float, default 0.3
            Dropout rate for regularization
        learning_rate : float, default 0.001
            Learning rate for optimizer
        batch_size : int, default 32
            Batch size for training
        epochs : int, default 100
            Maximum number of training epochs
        early_stopping : bool, default True
            Whether to use early stopping
        patience : int, default 10
            Patience for early stopping
        random_state : int, default 42
            Random seed for reproducibility
        use_torch : bool, default True
            Use PyTorch if available, otherwise TensorFlow
        """
        super().__init__()
        
        if not TORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Deep models require either PyTorch or TensorFlow. "
                "Install with: pip install torch or pip install tensorflow"
            )
        
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.classes_ = None
        self._estimator_type = 'classifier'
        self.history_ = None
    
    def _build_torch_model(self, n_features: int, n_classes: int) -> "nn.Module":
        """Build PyTorch neural network."""
        layers = []
        input_size = n_features
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, n_classes))
        
        return nn.Sequential(*layers)
    
    def _build_tf_model(self, n_features: int, n_classes: int):
        """Build TensorFlow/Keras neural network."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_features,)))
        
        for hidden_size in self.hidden_layers:
            model.add(keras.layers.Dense(hidden_size, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(self.dropout))
        
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'DeepFaciesClassifier':
        """
        Fit the deep learning classifier.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets (facies labels)
            
        Returns
        -------
        DeepFaciesClassifier
            Self for method chaining
        """
        X, y = self._validate_fit_input(X, y)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        n_classes = len(self.classes_)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        if self.use_torch:
            self._fit_torch(X_train, y_train, X_val, y_val, n_classes)
        else:
            self._fit_tf(X_train, y_train, X_val, y_val, n_classes)
        
        return self
    
    def _fit_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int
    ):
        """Train PyTorch model."""
        if torch is None:
            raise ImportError("PyTorch is required for this method")
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        n_features = X_train.shape[1]
        self.model = self._build_torch_model(n_features, n_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_dataset = WellLogDataset(X_train, y_train.astype(np.int64))
        val_dataset = WellLogDataset(X_val, y_val.astype(np.int64))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.long())
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y.long()).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.history_ = history
        logger.info(f"Training completed. Best validation accuracy: {max(history['val_acc']):.4f}")
    
    def _fit_tf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int
    ):
        """Train TensorFlow/Keras model."""
        if keras is None:
            raise ImportError("TensorFlow is required for this method")
        
        n_features = X_train.shape[1]
        self.model = self._build_tf_model(n_features, n_classes)
        
        callbacks = []
        if self.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        self.history_ = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'val_acc': history.history['val_accuracy']
        }
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict facies classes.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Predicted facies labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_torch:
            return self._predict_torch(X_scaled)
        else:
            return self._predict_tf(X_scaled)
    
    def _predict_torch(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict using PyTorch model."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.numpy()
        
        return self.label_encoder.inverse_transform(predictions)
    
    def _predict_tf(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict using TensorFlow model."""
        probabilities = self.model.predict(X_scaled, verbose=0)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.label_encoder.inverse_transform(predicted_indices)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_torch:
            self.model.eval()
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1).numpy()
        else:
            probabilities = self.model.predict(X_scaled, verbose=0)
        
        return probabilities


class DeepPropertyPredictor(BaseEstimator):
    """
    Deep neural network for petrophysical property prediction.
    
    Multi-layer perceptron for regression tasks like permeability and
    porosity prediction from well log data.
    
    Example:
        >>> from geosuite.ml import DeepPropertyPredictor
        >>> 
        >>> model = DeepPropertyPredictor(
        ...     hidden_layers=[128, 64],
        ...     dropout=0.2
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        random_state: int = 42,
        use_torch: bool = True
    ):
        """
        Initialize deep property predictor.
        
        Parameters
        ----------
        hidden_layers : list of int, default [128, 64]
            Number of neurons in each hidden layer
        dropout : float, default 0.2
            Dropout rate for regularization
        learning_rate : float, default 0.001
            Learning rate for optimizer
        batch_size : int, default 32
            Batch size for training
        epochs : int, default 100
            Maximum number of training epochs
        early_stopping : bool, default True
            Whether to use early stopping
        patience : int, default 10
            Patience for early stopping
        random_state : int, default 42
            Random seed for reproducibility
        use_torch : bool, default True
            Use PyTorch if available, otherwise TensorFlow
        """
        super().__init__()
        
        if not TORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Deep models require either PyTorch or TensorFlow. "
                "Install with: pip install torch or pip install tensorflow"
            )
        
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self._estimator_type = 'regressor'
        self.history_ = None
    
    def _build_torch_model(self, n_features: int) -> "nn.Module":
        """Build PyTorch regression model."""
        layers = []
        input_size = n_features
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        
        return nn.Sequential(*layers)
    
    def _build_tf_model(self, n_features: int):
        """Build TensorFlow regression model."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_features,)))
        
        for hidden_size in self.hidden_layers:
            model.add(keras.layers.Dense(hidden_size, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(self.dropout))
        
        model.add(keras.layers.Dense(1))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'DeepPropertyPredictor':
        """
        Fit the deep learning regressor.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets (property values)
            
        Returns
        -------
        DeepPropertyPredictor
            Self for method chaining
        """
        X, y = self._validate_fit_input(X, y)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled,
            test_size=0.2,
            random_state=self.random_state
        )
        
        if self.use_torch:
            self._fit_torch(X_train, y_train, X_val, y_val)
        else:
            self._fit_tf(X_train, y_train, X_val, y_val)
        
        return self
    
    def _fit_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train PyTorch regression model."""
        if torch is None:
            raise ImportError("PyTorch is required for this method")
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        n_features = X_train.shape[1]
        self.model = self._build_torch_model(n_features)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_dataset = WellLogDataset(X_train, y_train.astype(np.float32))
        val_dataset = WellLogDataset(X_val, y_val.astype(np.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.history_ = history
        logger.info(f"Training completed. Best validation MAE: {min(history['val_mae']):.4f}")
    
    def _fit_tf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train TensorFlow regression model."""
        if keras is None:
            raise ImportError("TensorFlow is required for this method")
        
        n_features = X_train.shape[1]
        self.model = self._build_tf_model(n_features)
        
        callbacks = []
        if self.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        self.history_ = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'val_mae': history.history['val_mae']
        }
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict property values.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Predicted property values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_X.transform(X)
        
        if self.use_torch:
            self.model.eval()
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                predictions_scaled = self.model(X_tensor).squeeze().numpy()
        else:
            predictions_scaled = self.model.predict(X_scaled, verbose=0).ravel()
        
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        return predictions

