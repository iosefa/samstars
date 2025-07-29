import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from utils.utils import _write_geotiff


class ClassifiedImage:
    """Container for the classifier outputs and metadata."""

    def __init__(self, classified, confusion_matrix, report, shap_values,
                 transform, crs, params):
        self.classified = classified
        self.confusion_matrix = confusion_matrix
        self.report = report
        self.shap_values = shap_values
        self.transform = transform
        self.crs = crs
        self.params = params

    # ------------------------------------------------------------------
    def write_geotiff(self, output_path: str):
        """Save *classified* to *output_path* as a GeoTIFF."""
        _write_geotiff(self.classified, output_path, self.crs, self.transform)


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _report_and_drop_nans(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Prints how many rows/cols contain NaNs and returns a cleaned copy."""
    non_geom = df.drop(columns=[df.geometry.name], errors="ignore")
    nan_rows = non_geom.isna().any(axis=1)

    if nan_rows.any():
        cols_with_nan = non_geom.columns[non_geom.isna().any()].tolist()
        print(f"[{df_name}] Dropping {nan_rows.sum()} rows with NaNs.")
        print(f"[{df_name}] Columns containing NaNs: {cols_with_nan}")
        return df.loc[~nan_rows].copy()
    return df.copy()


def _coerce_acceptable(value):
    """Ensure *value* is returned as a *list[int]* (may be empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(v) for v in value if str(v).isdigit()]
    # assume str / iterable of chars  -> split on comma
    return [int(tok) for tok in str(value).split(',') if tok.strip().isdigit()]


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def classify(
    segments: "gpd.GeoDataFrame",
    training_classes: pd.DataFrame,
    acceptable_classes_gdf: "gpd.GeoDataFrame | None" = None,
    *,
    method: str = "rf",
    test_size: float = 0.2,
    compute_reports: bool = False,
    compute_shap: bool = False,
    sample_shap: bool = False,
    **kwargs,
):
    """Train an RF / MLP on *training_classes* and label *segments*.

    Parameters
    ----------
    segments : GeoDataFrame
        Segments to predict.  May or may not already include a *feature_class*.
    training_classes : DataFrame
        Training samples with columns `feature_class`, `geometry`, `segment_id` +
        feature columns.
    acceptable_classes_gdf : GeoDataFrame, optional
        Polygons with an `acceptable_classes` column containing comma‑separated
        or list‐like class codes.  Predictions falling inside a polygon are
        filtered to these classes; if none overlap, we fall back to the raw
        prediction.
    method : {"rf", "mlp"}
    test_size : float
    compute_reports : bool
    compute_shap : bool
    sample_shap : bool
    kwargs : dict
        Extra keyword args forwarded to the sklearn classifier.

    Returns
    -------
    ClassifiedImage
    """

    # 1.  Clean NaNs and align indices -----------------------------------
    training_classes = _report_and_drop_nans(training_classes, "TRAINING").reset_index(drop=True)
    segments = _report_and_drop_nans(segments, "SEGMENTS").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2.  Train / test split and scaling
    # ------------------------------------------------------------------
    X = training_classes.drop(["feature_class", "geometry", "segment_id"], axis=1)
    y = training_classes["feature_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 3.  Initialise and train classifier
    # ------------------------------------------------------------------
    if method == "rf":
        classifier = RandomForestClassifier(**kwargs)
    elif method == "mlp":
        classifier = MLPClassifier(**kwargs)
    else:
        raise ValueError("Unsupported classifier: choose 'rf' or 'mlp'.")

    classifier.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4.  Optional SHAP explanation
    # ------------------------------------------------------------------
    shap_values = None
    if compute_shap:
        if isinstance(classifier, RandomForestClassifier):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_train)
        else:  # MLP
            bg = shap.sample(X_train, 500, random_state=42) if sample_shap else X_train
            explainer = shap.KernelExplainer(classifier.predict_proba, bg)
            shap_values = explainer.shap_values(bg)

    # ------------------------------------------------------------------
    # 5.  Optional performance report
    # ------------------------------------------------------------------
    report = cm = None
    if compute_reports:
        y_pred_test = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test)

    # ------------------------------------------------------------------
    # 6.  Predict for *segments*
    # ------------------------------------------------------------------
    X_pred = scaler.transform(
        segments.drop(["feature_class", "geometry", "segment_id"], axis=1, errors="ignore")
    )

    y_pred_all = np.empty(X_pred.shape[0], dtype=object)
    prediction_margin = np.empty(X_pred.shape[0], dtype=float)

    class_labels_int = classifier.classes_.astype(int)

    for idx, segment in segments.iterrows():
        # --------------------------------------------------------------
        # Retrieve acceptable classes for this segment (if any)
        # --------------------------------------------------------------
        acceptable_classes = None
        if acceptable_classes_gdf is not None:
            intersects = acceptable_classes_gdf[acceptable_classes_gdf.intersects(segment.geometry)]
            if not intersects.empty:
                acceptable_classes = intersects.iloc[0]["acceptable_classes"]

        acceptable_list = _coerce_acceptable(acceptable_classes)
        acceptable_set = set(acceptable_list)

        # --------------------------------------------------------------
        # Probability vector
        # --------------------------------------------------------------
        proba_vec = classifier.predict_proba([X_pred[idx]])[0]
        proba_df = pd.DataFrame([proba_vec], columns=class_labels_int)

        keep_cols = [c for c in proba_df.columns if c in acceptable_set] if acceptable_set else []

        if keep_cols:
            proba_df_filtered = proba_df[keep_cols]
            y_pred_all[idx] = proba_df_filtered.idxmax(axis=1).values[0]
            probs = proba_df_filtered.values[0]
        else:
            y_pred_all[idx] = proba_df.idxmax(axis=1).values[0]
            probs = proba_df.values[0]

        # --------------------------------------------------------------
        # Prediction margin (robust to length-1)
        # --------------------------------------------------------------
        if probs.size >= 2:
            k = probs.size - 2
            top2 = np.partition(probs, k)[-2:]
            prediction_margin[idx] = top2[1] - top2[0]
        else:
            prediction_margin[idx] = float(probs[0])

    # ------------------------------------------------------------------
    # 7.  Assemble output GeoDataFrame
    # ------------------------------------------------------------------
    segments["predicted_class"] = y_pred_all
    segments["prediction_margin"] = prediction_margin.astype(float)

    # Ensure numeric columns have proper dtypes
    for col in segments.columns:
        if col != segments.geometry.name:
            if np.issubdtype(segments[col].dtype, np.integer):
                segments[col] = segments[col]
            elif np.issubdtype(segments[col].dtype, np.floating):
                segments[col] = segments[col].astype(float)

    return ClassifiedImage(
        segments, cm, report, shap_values, None, None, classifier.get_params()
    )