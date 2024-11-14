from django.db import models
from django.db.models import Avg, Count, F, Sum
from django.utils import timezone
from datetime import timedelta
import numpy as np
from typing import Tuple, Optional
import logging
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from shop.models import Product, Order

logger = logging.getLogger(__name__)

class OrderPredictorDNN:
    def __init__(self, model_dir='ml_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model_path = self.model_dir / 'order_predictor_dnn.h5'
        self.scaler_path = self.model_dir / 'feature_scaler.pkl'
        
        self.model = self._load_or_create_model()
        self.scaler = self._load_or_create_scaler()
        
        self.batch_size = 32
        self.epochs = 750
        
    def _create_dnn_model(self, input_dim):
        """Create a new DNN model with the specified architecture."""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

    def _load_or_create_model(self) -> tf.keras.Model:
        """Load existing model or create a new one if not found."""
        try:
            if self.model_path.exists():
                logger.info("Loading existing DNN model...")
                return load_model(self.model_path, custom_objects={'loss': 'mean_squared_error'})
            else:
                logger.info("Creating new DNN model...")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def _load_or_create_scaler(self) -> StandardScaler:
        """Load existing scaler or create a new one."""
        try:
            if self.scaler_path.exists():
                logger.info("Loading existing scaler...")
                return joblib.load(self.scaler_path)
            else:
                logger.info("Creating new scaler...")
                scaler = StandardScaler()
                scaler.n_samples_seen_ = 0 
                return scaler
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            scaler = StandardScaler()
            scaler.n_samples_seen_ = 0 
            return scaler

    def save_model(self):
        """Save the trained model and scaler."""
        try:
            if self.model is not None:
                self.model.save(self.model_path)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
            logger.info("Model and scaler saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def extract_features(self, user, product, reference_date=None) -> np.ndarray:
        """Extract relevant features for prediction."""
        try:
            reference_date = reference_date or timezone.now()
            all_orders = list(user.order_set.all().select_related('product').order_by('order_date'))
            product_orders = [o for o in all_orders if o.product_id == product.id]
            
            last_5_intervals = []
            last_5_quantities = []
            
            if len(product_orders) >= 2:
                for i in range(min(5, len(product_orders) - 1)):
                    interval = (product_orders[-(i+1)].order_date - product_orders[-(i+2)].order_date).days
                    last_5_intervals.append(interval)
                    last_5_quantities.append(product_orders[-(i+1)].quantity)
            
            last_5_intervals = (last_5_intervals + [0] * 5)[:5]
            last_5_quantities = (last_5_quantities + [0] * 5)[:5]
            
            total_orders = len(all_orders)
            product_orders_count = len(product_orders)
            
            if product_orders:
                first_order = product_orders[0].order_date
                last_order = product_orders[-1].order_date
                date_range = (last_order - first_order).days + 1
                product_frequency = product_orders_count / max(1, date_range)
                days_since_last = (reference_date - last_order).days
                avg_interval = date_range / max(1, product_orders_count - 1)
            else:
                product_frequency = 0
                days_since_last = 365
                avg_interval = 0
            
            quantities = [o.quantity for o in product_orders]
            values = [o.quantity * o.product.price for o in product_orders]
            
            avg_quantity = np.mean(quantities) if quantities else 0
            std_quantity = np.std(quantities) if len(quantities) > 1 else 0
            avg_value = np.mean(values) if values else 0
            std_value = np.std(values) if len(values) > 1 else 0
            
            thirty_days_ago = reference_date - timedelta(days=30)
            recent_orders = len([o for o in product_orders if o.order_date >= thirty_days_ago])
            recent_frequency = recent_orders / 30.0
            
            temporal_features = [
                float(reference_date.year),
                float(reference_date.month),
                float(reference_date.day),
                float(reference_date.weekday()),
                np.sin(2 * np.pi * reference_date.month / 12),
                np.cos(2 * np.pi * reference_date.month / 12),
                np.sin(2 * np.pi * reference_date.weekday() / 7),  
                np.cos(2 * np.pi * reference_date.weekday() / 7)
            ]
            
            features = np.array(
                last_5_intervals +  
                last_5_quantities +  
                [
                    float(product_orders_count),
                    float(total_orders),
                    product_frequency,
                    days_since_last,
                    avg_interval,
                    avg_quantity,
                    std_quantity,
                    float(avg_value),
                    float(std_value),
                    recent_frequency
                ] +
                temporal_features
            )
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
            
    def update_model(self, user):
        """Update the DNN model with new training data and calculate performance metrics."""
        try:
            X_train = []
            y_train = []
            
            orders = list(user.order_set.all().order_by('order_date'))
            
            for i in range(len(orders) - 1):
                current_order = orders[i]
                next_order = orders[i + 1]
                
                features = self.extract_features(
                    user, 
                    current_order.product, 
                    reference_date=current_order.order_date
                )
                X_train.append(features[0])
                
                days_until_next = (next_order.order_date - current_order.order_date).days
                y_train.append(min(days_until_next, 90))
            
            if X_train and y_train:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                if self.model is None:
                    input_dim = X_train.shape[1]
                    self.model = self._create_dnn_model(input_dim)
                
                if self.scaler.n_samples_seen_ == 0:
                    self.scaler.fit(X_train)
                X_train_scaled = self.scaler.transform(X_train)
                
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                history = self.model.fit(
                    X_train_scaled, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=0.2,
                    #callbacks=[early_stopping],
                    verbose=0
                )

                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                
                self.save_model()
                print("DNN model updated successfully")

                # Calculate performance metrics
                y_pred = self.model.predict(X_train_scaled, verbose=0).flatten()
                mae = mean_absolute_error(y_train, y_pred)
                mse = mean_squared_error(y_train, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_train, y_pred)

                print(f"Training Metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise

    def predict_next_order(self, user) -> Tuple[Optional[Product], Optional[timezone.datetime]]:
        """Predict the next order using the DNN model."""
        try:
            if user.order_set.count() > 1:
                self.update_model(user)
            
            frequent_products = (user.order_set.values('product')
                               .annotate(order_count=Count('id'))
                               .order_by('-order_count')[:5])
            
            if not frequent_products:
                return None, None
            
            best_product = None
            min_days = float('inf')
            
            for product_info in frequent_products:
                product = Product.objects.get(id=product_info['product'])
                features = self.extract_features(user, product)
                
                if self.scaler and self.model:
                    features_scaled = self.scaler.transform(features)
                    days_until_next = self.model.predict(features_scaled, verbose=0)[0][0]
                    if days_until_next < min_days:
                        min_days = days_until_next
                        best_product = product
            
            if best_product is None:
                return None, None
            
            next_order_date = timezone.now() + timedelta(days=int(min_days))
            return best_product, next_order_date
        
        except Exception as e:
            logger.error(f"Error predicting next order: {str(e)}")
            return None, None


def predict_next_order(user) -> Tuple[Optional[Product], Optional[timezone.datetime]]:
    """Wrapper function to predict next order."""
    try:
        predictor = OrderPredictorDNN()
        return predictor.predict_next_order(user)
    except Exception as e:
        logger.error(f"Error predicting next order: {str(e)}")
        return None, None