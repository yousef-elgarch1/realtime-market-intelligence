import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
import os

logger = logging.getLogger(__name__)

class MLModelTracker:
    """MLflow integration for tracking ML experiments and model versions"""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", 
                 experiment_name: str = "BCG_X_Market_Intelligence"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = None
        
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient()
            
            logger.info(f"✅ MLflow tracking initialized: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MLflow: {e}")
            self.client = None
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run"""
        if not self.client:
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"🚀 Started MLflow run: {run.info.run_id}")
            return run.info.run_id
        except Exception as e:
            logger.error(f"❌ Failed to start MLflow run: {e}")
            return None
    
    def log_sentiment_model_metrics(self, run_id: str, metrics: Dict[str, float],
                                  model_params: Dict[str, Any] = None):
        """Log sentiment analysis model metrics"""
        try:
            with mlflow.start_run(run_id=run_id):
                # Log parameters
                if model_params:
                    for param, value in model_params.items():
                        mlflow.log_param(param, value)
                
                # Log metrics
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                
                # Log model type
                mlflow.log_param("model_type", "sentiment_analysis")
                mlflow.log_param("model_framework", "transformers")
                
                logger.info(f"✅ Logged sentiment model metrics to run {run_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to log sentiment metrics: {e}")
    
    def log_price_prediction_model(self, run_id: str, model, model_params: Dict,
                                 training_history: Dict, validation_metrics: Dict,
                                 prediction_horizon: int):
        """Log LSTM price prediction model with full details"""
        try:
            with mlflow.start_run(run_id=run_id):
                # Log model parameters
                mlflow.log_param("model_type", "lstm_price_prediction")
                mlflow.log_param("prediction_horizon_hours", prediction_horizon)
                mlflow.log_param("framework", "tensorflow")
                
                for param, value in model_params.items():
                    mlflow.log_param(param, value)
                
                # Log training history
                if 'loss' in training_history:
                    for epoch, loss in enumerate(training_history['loss']):
                        mlflow.log_metric("training_loss", loss, step=epoch)
                
                if 'val_loss' in training_history:
                    for epoch, val_loss in enumerate(training_history['val_loss']):
                        mlflow.log_metric("validation_loss", val_loss, step=epoch)
                
                # Log validation metrics
                for metric, value in validation_metrics.items():
                    mlflow.log_metric(f"final_{metric}", value)
                
                # Log model
                mlflow.tensorflow.log_model(
                    model, 
                    f"lstm_model_{prediction_horizon}h",
                    registered_model_name=f"crypto_price_predictor_{prediction_horizon}h"
                )
                
                logger.info(f"✅ Logged LSTM model ({prediction_horizon}h) to run {run_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to log LSTM model: {e}")
    
    def log_prediction_batch(self, predictions: List[Dict], 
                           model_version: str = "latest"):
        """Log a batch of predictions for monitoring"""
        try:
            run_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Log prediction metadata
                mlflow.log_param("prediction_type", "batch_crypto_prices")
                mlflow.log_param("model_version", model_version)
                mlflow.log_param("num_predictions", len(predictions))
                mlflow.log_param("timestamp", datetime.utcnow().isoformat())
                
                # Aggregate prediction metrics
                if predictions:
                    confidences = [p.get('confidence', 0) for p in predictions 
                                 if 'confidence' in p]
                    if confidences:
                        mlflow.log_metric("avg_prediction_confidence", np.mean(confidences))
                        mlflow.log_metric("min_prediction_confidence", np.min(confidences))
                        mlflow.log_metric("max_prediction_confidence", np.max(confidences))
                
                # Log predictions as artifact
                predictions_df = pd.DataFrame(predictions)
                predictions_df.to_csv("predictions_batch.csv", index=False)
                mlflow.log_artifact("predictions_batch.csv")
                
                logger.info(f"✅ Logged {len(predictions)} predictions")
                
        except Exception as e:
            logger.error(f"❌ Failed to log predictions: {e}")
    
    def get_best_model(self, model_name: str, metric: str = "validation_loss",
                      ascending: bool = True) -> Dict:
        """Get the best model version based on a metric"""
        try:
            # Search for runs with the model
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_type = '{model_name}'",
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )
            
            if not runs.empty:
                best_run = runs.iloc[0]
                
                return {
                    'run_id': best_run['run_id'],
                    'model_uri': f"runs:/{best_run['run_id']}/{model_name}",
                    'metrics': {col.replace('metrics.', ''): best_run[col] 
                              for col in best_run.index if col.startswith('metrics.')},
                    'params': {col.replace('params.', ''): best_run[col] 
                             for col in best_run.index if col.startswith('params.')},
                    'timestamp': best_run.get('start_time', 'unknown')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get best model: {e}")
            return None
    
    def log_model_performance(self, model_type: str, symbol: str,
                            actual_prices: List[float], predicted_prices: List[float],
                            timestamps: List[datetime]):
        """Log real-world model performance"""
        try:
            run_name = f"performance_{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d')}"
            
            with mlflow.start_run(run_name=run_name):
                # Calculate performance metrics
                actual_np = np.array(actual_prices)
                predicted_np = np.array(predicted_prices)
                
                # Mean Absolute Error
                mae = np.mean(np.abs(actual_np - predicted_np))
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((actual_np - predicted_np) / actual_np)) * 100
                
                # Root Mean Square Error
                rmse = np.sqrt(np.mean((actual_np - predicted_np) ** 2))
                
                # Directional Accuracy (did we predict the direction correctly?)
                if len(actual_prices) > 1:
                    actual_direction = np.diff(actual_np) > 0
                    predicted_direction = np.diff(predicted_np) > 0
                    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
                else:
                    directional_accuracy = 0
                
                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("directional_accuracy", directional_accuracy)
                
                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("crypto_symbol", symbol)
                mlflow.log_param("evaluation_period", f"{len(actual_prices)} data points")
                
                # Create performance dataframe and log as artifact
                performance_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'actual_price': actual_prices,
                    'predicted_price': predicted_prices,
                    'absolute_error': np.abs(actual_np - predicted_np),
                    'percentage_error': np.abs((actual_np - predicted_np) / actual_np) * 100
                })
                
                performance_df.to_csv(f"performance_{symbol}.csv", index=False)
                mlflow.log_artifact(f"performance_{symbol}.csv")
                
                logger.info(f"✅ Logged performance for {model_type} on {symbol}: "
                          f"MAE={mae:.4f}, MAPE={mape:.2f}%, Directional={directional_accuracy:.1f}%")
                
                return {
                    'mae': mae,
                    'mape': mape,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to log model performance: {e}")
            return None
    
    def get_model_leaderboard(self, model_type: str = None) -> pd.DataFrame:
        """Get a leaderboard of model performance"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            filter_string = ""
            if model_type:
                filter_string = f"params.model_type = '{model_type}'"
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                return pd.DataFrame()
            
            # Select relevant columns for leaderboard
            leaderboard_cols = ['run_id', 'start_time', 'status']
            
            # Add parameter columns
            param_cols = [col for col in runs.columns if col.startswith('params.')]
            leaderboard_cols.extend(param_cols)
            
            # Add metric columns
            metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
            leaderboard_cols.extend(metric_cols)
            
            # Filter to existing columns
            available_cols = [col for col in leaderboard_cols if col in runs.columns]
            
            leaderboard = runs[available_cols].copy()
            
            # Clean up column names
            leaderboard.columns = [col.replace('params.', '').replace('metrics.', '') 
                                 for col in leaderboard.columns]
            
            return leaderboard.head(20)  # Top 20 runs
            
        except Exception as e:
            logger.error(f"❌ Failed to get model leaderboard: {e}")
            return pd.DataFrame()
    
    def cleanup_old_runs(self, days_to_keep: int = 30):
        """Clean up old MLflow runs to save space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
            
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attribute.start_time < {cutoff_timestamp}"
            )
            
            deleted_count = 0
            for _, run in runs.iterrows():
                try:
                    self.client.delete_run(run['run_id'])
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete run {run['run_id']}: {e}")
            
            logger.info(f"🧹 Cleaned up {deleted_count} old MLflow runs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old runs: {e}")
            return 0

# Global instance
ml_tracker = MLModelTracker()