"""
Machine Learning clustering and dimensionality reduction pipeline.
Includes K-Means, Agglomerative, DBSCAN, Evaluation Metrics (Elbow, Silhouette), and PCA.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

def scale_features(X: pd.DataFrame, scaler_type: str = "StandardScaler") -> Tuple[np.ndarray, Any]:
    """Scales feature matrix using the specified scaler."""
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def compute_elbow_and_silhouette(
    X_scaled: np.ndarray, 
    k_min: int = 2, 
    k_max: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """Computes WCSS (Inertia) and Silhouette Scores for a range of k values."""
    k_values = list(range(k_min, min(k_max + 1, len(X_scaled))))
    wcss = []
    silhouette_scores = []
    
    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        try:
            sil = silhouette_score(X_scaled, labels)
            silhouette_scores.append(round(sil, 4))
        except Exception:
            silhouette_scores.append(0.0)
            
    # Simple heuristic to identify elbow (maximum distance to chord line)
    # or peak silhouette score
    best_k_sil = k_values[int(np.argmax(silhouette_scores))]
    
    return {
        "k_values": k_values,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "best_k_silhouette": best_k_sil
    }

def fit_clustering_model(
    algorithm: str,
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Fits chosen clustering algorithm on scaled data.
    Returns:
        (cluster_labels, model_instance, evaluation_metrics)
    """
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "Hierarchical (Agglomerative)":
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X_scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
        
    # Evaluate clusters (excluding noise points -1 in DBSCAN for silhouette calculation)
    valid_mask = labels != -1
    unique_clusters = len(set(labels[valid_mask]))
    
    metrics = {
        "unique_clusters": unique_clusters,
        "noise_points": int(np.sum(labels == -1)) if -1 in labels else 0
    }
    
    if unique_clusters >= 2 and np.sum(valid_mask) > unique_clusters:
        try:
            metrics["silhouette"] = round(silhouette_score(X_scaled[valid_mask], labels[valid_mask]), 4)
            metrics["davies_bouldin"] = round(davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]), 4)
            metrics["calinski_harabasz"] = round(calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask]), 2)
        except Exception:
            metrics["silhouette"] = None
            metrics["davies_bouldin"] = None
            metrics["calinski_harabasz"] = None
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
        
    if hasattr(model, 'inertia_'):
        metrics["inertia"] = round(model.inertia_, 2)
        
    return labels, model, metrics

def apply_pca_reduction(X_scaled: np.ndarray) -> Dict[str, Any]:
    """Performs PCA projection to 2D and 3D spaces for visualization."""
    n_features = X_scaled.shape[1]
    
    # 2D PCA
    pca_2d = PCA(n_components=min(2, n_features))
    coords_2d = pca_2d.fit_transform(X_scaled)
    var_2d = [round(float(v) * 100, 2) for v in pca_2d.explained_variance_ratio_]
    
    # 3D PCA if 3 or more features
    if n_features >= 3:
        pca_3d = PCA(n_components=min(3, n_features))
        coords_3d = pca_3d.fit_transform(X_scaled)
        var_3d = [round(float(v) * 100, 2) for v in pca_3d.explained_variance_ratio_]
    else:
        coords_3d = None
        var_3d = []
        
    return {
        "coords_2d": coords_2d,
        "var_2d": var_2d,
        "coords_3d": coords_3d,
        "var_3d": var_3d
    }
