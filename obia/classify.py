import shap, warnings, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.utils import _write_geotiff, label_segments
import geopandas as gpd
from xgboost import XGBClassifier


class ClassifiedImage:
    """Container for the classifier outputs and metadata."""
    # (unchanged) -------------------------------------------------------
    def __init__(self, classified, confusion_matrix, report, shap_values,
                 transform, crs, params):
        self.classified = classified
        self.confusion_matrix = confusion_matrix
        self.report = report
        self.shap_values = shap_values
        self.transform = transform
        self.crs = crs
        self.params = params

    def write_geotiff(self, output_path: str):
        _write_geotiff(self.classified, output_path, self.crs, self.transform)


# -------------- helpers (unchanged) -----------------------------------
def _report_and_drop_nans(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    non_geom = df.drop(columns=[df.geometry.name], errors="ignore")
    nan_rows = non_geom.isna().any(axis=1)
    if nan_rows.any():
        cols_with_nan = non_geom.columns[non_geom.isna().any()].tolist()
        print(f"[{df_name}] Dropping {nan_rows.sum()} rows with NaNs.")
        print(f"[{df_name}] Columns containing NaNs: {cols_with_nan}")
        return df.loc[~nan_rows].copy()
    return df.copy()

def _coerce_acceptable(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(v) for v in value if str(v).isdigit()]
    return [int(tok) for tok in str(value).split(',') if tok.strip().isdigit()]


# ---------------------------- main ------------------------------------
def classify(
    segments: "gpd.GeoDataFrame",
    training_classes: pd.DataFrame,
    acceptable_classes_gdf: "gpd.GeoDataFrame | None" = None,
    *,
    method: str = "rf",          # ← now accepts "xgb"
    test_size: float = 0.2,
    compute_reports: bool = False,
    compute_shap: bool = False,
    sample_shap: bool = False,
    margin_threshold: float = 0.25,
    **kwargs,
):
    """
    Train a classifier on *training_classes* and label *segments*.

    Parameters
    ----------
    method : {"rf", "xgb", "mlp"}
        "xgb" uses XGBoost; requires `pip install xgboost`.
    ...
    """

    # 1. clean NaNs -----------------------------------------------------
    training_classes = _report_and_drop_nans(training_classes, "TRAINING").reset_index(drop=True)
    segments         = _report_and_drop_nans(segments,         "SEGMENTS").reset_index(drop=True)

    # 2. split & (optionally) scale ------------------------------------
    feat_cols = [c for c in training_classes.columns
                 if c not in ("feature_class", "geometry", "segment_id")]

    X = training_classes[feat_cols]
    y = training_classes["feature_class"]

    label_enc = None
    if method == "xgb":
        label_enc = LabelEncoder().fit(y)
        y = label_enc.transform(y)  # 0 … n-1

    if test_size is None or float(test_size) <= 0:
        X_tr, y_tr = X, y
        X_te, y_te = None, None
    else:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            if "least populated classes" in str(e) or "cannot be less than 2" in str(e):
                print("[classify] stratified split failed; falling back to unstratified split.")
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=None
                )
            else:
                raise

    scaler = StandardScaler().fit(X_tr) if method == "mlp" else None
    if scaler is not None:
        X_tr = scaler.transform(X_tr)
        if X_te is not None:
            X_te = scaler.transform(X_te)

    # 3. initialise classifier ----------------------------------------
    if method == "rf":
        defaults = dict(
            n_estimators=500,  # strong baseline
            max_depth=None,  # let trees grow; control with min_samples_leaf
            min_samples_split=2,
            min_samples_leaf=2,  # mild regularization vs leaf=1
            max_features="sqrt",
            bootstrap=True,
            oob_score=False,  # set True if you want OOB metrics
            class_weight="balanced_subsample",  # handles class imbalance
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        clf = RandomForestClassifier(**defaults)

    elif method == "xgb":
        defaults = dict(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        clf = XGBClassifier(**defaults)

    elif method == "mlp":
        print("METHOD = MLP")
        defaults = dict(
            hidden_layer_sizes=(64, 32),  # compact, effective
            activation="relu",
            solver="adam",
            alpha=1e-4,  # L2 regularization
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            shuffle=True,
            random_state=42,
            verbose=False,
        )
        defaults.update(kwargs)
        clf = MLPClassifier(**defaults)

    else:
        raise ValueError("method must be 'rf', 'xgb', or 'mlp'.")

    # 4. train ----------------------------------------------------------
    clf.fit(X_tr, y_tr)

    # 5. SHAP (optional) -----------------------------------------------
    shap_vals = None
    if compute_shap:
        if method in ("rf", "xgb"):
            bg = shap.sample(X_tr, 500, random_state=42) if sample_shap else X_tr
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(bg)
        else:
            bg = shap.sample(X_tr, 500, random_state=42) if sample_shap else X_tr
            explainer = shap.KernelExplainer(clf.predict_proba, bg)
            shap_vals = explainer.shap_values(bg)

    # 6. reports --------------------------------------------------------
    cm = rpt = None
    if compute_reports and X_te is not None and y_te is not None:
        y_pred = clf.predict(X_te)
        cm     = confusion_matrix(y_te, y_pred)
        rpt    = classification_report(y_te, y_pred)

    # 7. predict for *segments* ----------------------------------------
    X_pred_raw = segments[feat_cols]
    X_pred = scaler.transform(X_pred_raw) if scaler is not None else X_pred_raw.values
    proba_all = clf.predict_proba(X_pred)

    if label_enc is not None:
        class_int = label_enc.inverse_transform(np.arange(proba_all.shape[1]))
    else:
        class_int = clf.classes_.astype(int)

    n = X_pred.shape[0]
    y_pred_final       = np.empty(n, dtype=object)
    y_pred_raw         = np.empty(n, dtype=object)
    pred_margin        = np.empty(n, dtype=float)
    reclassified_flag  = np.zeros(n, dtype=bool)

    for idx, seg in segments.iterrows():
        # Full probability vector and labels (no masking)
        probs_full = proba_all[idx]
        labels     = class_int

        # Raw prediction from full vector
        best_idx = int(np.argmax(probs_full))
        y_raw = labels[best_idx]
        y_pred_raw[idx] = y_raw

        # Compute top-2 margin from full vector
        if probs_full.size >= 2:
            k = probs_full.size - 2
            top2 = np.partition(probs_full, k)[-2:]
            margin = float(top2[1] - top2[0])
        else:
            margin = float(probs_full[0])
        pred_margin[idx] = margin

        # Optionally apply acceptable-classes rule ONLY if low-confidence
        y_final = y_raw
        if acceptable_classes_gdf is not None and margin < float(margin_threshold):
            acceptable = None
            hit = acceptable_classes_gdf[acceptable_classes_gdf.intersects(seg.geometry)]
            if not hit.empty:
                acceptable = hit.iloc[0].get("acceptable_classes", None)
            allowed_set = set(_coerce_acceptable(acceptable))

            # If the raw prediction is implausible, choose the highest-prob among allowed
            if allowed_set and (y_raw not in allowed_set):
                mask = np.isin(labels, list(allowed_set))
                if np.any(mask):
                    # Choose the best within allowed set; if runner-up is allowed, this will often pick it
                    best_allowed_idx = int(np.argmax(probs_full[mask]))
                    y_final = labels[mask][best_allowed_idx]
                    reclassified_flag[idx] = True
                # else: keep y_final as raw (no allowed alternative present)

        y_pred_final[idx] = y_final

    # 8. assemble -------------------------------------------------------
    segments["predicted_class_raw"] = y_pred_raw
    segments["predicted_class"]     = y_pred_final
    segments["prediction_margin"]   = pred_margin.astype(float)
    segments["reclassified"]        = reclassified_flag.astype(bool)

    result = ClassifiedImage(
        segments, cm, rpt, shap_vals,
        transform=None, crs=None,
        params=clf.get_params()
    )
    result.model_  = clf
    result.scaler_ = scaler
    return result
