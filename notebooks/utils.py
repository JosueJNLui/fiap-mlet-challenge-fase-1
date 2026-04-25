import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_recall_curve


NUMERIC_COLS = ("tenure", "MonthlyCharges", "TotalCharges")

LTV_DEFAULT = 500
COST_DEFAULT = 100

YES_NO_COLS = (
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
)

ONE_HOT_COLS = ("InternetService", "Contract", "PaymentMethod")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza determinística sem feature engineering. Replica o pipeline de models-simpler."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = df[df["TotalCharges"] != " "].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    df = df.replace("No internet service", "No").replace("No phone service", "No")

    for col in YES_NO_COLS:
        df[col] = df[col].eq("Yes").astype("int64")
    df["gender"] = df["gender"].eq("Female").astype("int64")

    if "Churn" in df.columns:
        df = df.rename(columns={"Churn": "target"})
    if df["target"].dtype == object:
        df["target"] = df["target"].map({"Yes": 1, "No": 0}).astype("int64")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding sem drop_first. Resulta em 26 features para casar com input_shape da ANN."""
    df = pd.get_dummies(df, columns=list(ONE_HOT_COLS), drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int64")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return encode_features(clean_dataset(df))


def split_data(df_encoded: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2,
               random_state: int = 42, target_col: str = "target"):
    """Sempre retorna 6-tuple. Baselines desempacotam com `_` para val."""
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_numeric(X_train, X_val, X_test, num_cols=NUMERIC_COLS):
    """MinMax scaling. Fit no train, transform em val/test. Retorna scaler para persistência."""
    cols = list(num_cols)
    scaler = MinMaxScaler()
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_val, X_test, scaler


def pos_weight_balanced(y_train):
    """Calcula pos_weight = n_neg / n_pos para BCEWithLogitsLoss (equivalente a class_weight='balanced').

    Retorna torch.Tensor de shape (1,). Import local de torch para manter utils.py leve.
    """
    import torch
    y = np.asarray(y_train)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def lucro_liquido(y_true, y_pred, ltv=LTV_DEFAULT, cost=COST_DEFAULT):
    """Lucro líquido = TP*(ltv-cost) - FP*cost - FN*ltv. Mesma fórmula do evaluate_model."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp * (ltv - cost) - fp * cost - fn * ltv


def find_threshold_max_profit(y_true, proba, ltv=LTV_DEFAULT, cost=COST_DEFAULT, grid=None):
    """Threshold que maximiza lucro líquido. Varre grid de 0.01 a 0.99 (passo 0.01) por padrão."""
    if grid is None:
        grid = np.arange(0.01, 1.00, 0.01)
    proba = np.asarray(proba)
    best_thr = 0.5
    best_profit = -np.inf
    for thr in grid:
        profit = lucro_liquido(y_true, (proba >= thr).astype(int), ltv=ltv, cost=cost)
        if profit > best_profit:
            best_profit = profit
            best_thr = float(thr)
    return best_thr, best_profit


def find_threshold_max_f1(y_true, proba):
    """Threshold que maximiza F1 da classe positiva. Usa precision_recall_curve para grid eficiente."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # f1 = 2*p*r/(p+r); cuidado com divisão por zero
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    # precision_recall_curve retorna len(thr) = len(prec)-1; alinhar
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thr[idx]), float(f1[idx])


def find_threshold_min_recall(y_true, proba, recall_target=0.75):
    """Maior threshold que ainda atinge recall >= recall_target (maximiza precisão sob restrição)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    grid = np.arange(0.01, 1.00, 0.01)
    candidatos = []
    for thr in grid:
        pred = (proba >= thr).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if rec >= recall_target:
            candidatos.append((float(thr), rec))
    if not candidatos:
        return 0.0, 0.0  # Nenhum threshold atinge o alvo; usar 0 (tudo positivo)
    # Maior threshold que satisfaz a restrição => maior precisão
    return max(candidatos, key=lambda x: x[0])


def calibrate_probas(proba_val, y_val, *probas_to_transform):
    """Calibração isotônica fitada no val. Retorna probas calibradas + calibrator.

    Uso: proba_val_cal, proba_test_cal, calibrator = calibrate_probas(proba_val, y_val, proba_test)
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_val, y_val)
    transformed = [calibrator.transform(np.asarray(p)) for p in probas_to_transform]
    proba_val_cal = calibrator.transform(np.asarray(proba_val))
    return (proba_val_cal, *transformed, calibrator)
