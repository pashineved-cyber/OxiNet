import numpy as np
import pandas as pd

def preprocess_hrv_data(hrv_data: np.ndarray) -> np.ndarray:
    """
    Preprocess HRV data by removing outliers and normalizing.
    
    Args:
        hrv_data (np.ndarray): Raw HRV data.
    
    Returns:
        np.ndarray: Processed HRV data.
    """
    # Remove outliers (example: z-score method)
    z_scores = (hrv_data - np.mean(hrv_data)) / np.std(hrv_data)
    filtered_hrv = hrv_data[np.abs(z_scores) < 3]
    return (filtered_hrv - np.min(filtered_hrv)) / (np.max(filtered_hrv) - np.min(filtered_hrv))

def multi_modal_feature_fusion(hrv_features: np.ndarray, spo2_features: np.ndarray) -> np.ndarray:
    """
    Fuse HRV and SpO2 features into a single feature vector.
    
    Args:
        hrv_features (np.ndarray): Processed HRV features.
        spo2_features (np.ndarray): Processed SpO2 features.
    
    Returns:
        np.ndarray: Combined feature vector.
    """
    return np.concatenate((hrv_features, spo2_features), axis=0)

def create_metadata_vectors(hrv_data: np.ndarray, demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Create metadata vectors from HRV data and demographic information.
    
    Args:
        hrv_data (np.ndarray): Processed HRV data.
        demographics (pd.DataFrame): DataFrame containing demographic information.
    
    Returns:
        pd.DataFrame: DataFrame containing metadata vectors.
    """
    metadata_vectors = demographics.copy()
    metadata_vectors['hrv_mean'] = np.mean(hrv_data)
    metadata_vectors['hrv_std'] = np.std(hrv_data)
    return metadata_vectors

def normalize_combined_inputs(hrv_data: np.ndarray, spo2_data: np.ndarray) -> np.ndarray:
    """
    Normalize combined HRV and SpO2 data.
    
    Args:
        hrv_data (np.ndarray): Processed HRV data.
        spo2_data (np.ndarray): Processed SpO2 data.
    
    Returns:
        np.ndarray: Normalized data.
    """
    combined_data = np.concatenate((hrv_data, spo2_data), axis=0)
    return (combined_data - np.min(combined_data)) / (np.max(combined_data) - np.min(combined_data))
