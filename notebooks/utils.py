from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # Tenure bucket
    bins = [-1, 12, 24, 48, np.inf]
    labels = ['0-12', '13-24', '25-48', '49+']
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # # Avg Charges per month
    df['avg_charges_per_month'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    # Charge vs EXpected
    df['charge_vs_expected'] = df['MonthlyCharges'] - df['avg_charges_per_month']
    df.drop(columns=['avg_charges_per_month'], inplace=True)

    # Num services
    service_list = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'InternetService']
    df['num_services'] = df[service_list].sum(axis=1)

    return df

def preprocessing(df: pd.DataFrame) -> tuple:

    # Limpeza inicial - Remover coluuna de ID
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True)

    # Converter para binário
    binary_cols = ['Partner', 'Dependents', 'PhoneService'
                   , 'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup'
                   , 'DeviceProtection', 'TechSupport', 'StreamingTV'
                   , 'StreamingMovies', 'MultipleLines'
                   ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
    
    # Tratar coluna InternetService
    df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 1, 'No': 0})

    # Tratar coluna TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']

    # Aplicar transformação Logarítmica
    df['TotalCharges'] = np.log1p(df['TotalCharges'])

    # Criação das Features
    df = create_features(df)

    # Remover colunas
    cols_to_drop = [
        'tenure', 
        'gender', 
        'SeniorCitizen', 
        'Dependents', 
        'PaperlessBilling'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # One-Hot Encoding usando get_dummies
    df_encoded = hot_encoding(df)

    return df_encoded

def hot_encoding(df: pd.DataFrame) -> pd.DataFrame:

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Colunas categóricas ({len(categorical_cols)}): {categorical_cols}\n")

    # Mostrar cardinalidade de cada coluna categórica
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} valores únicos → {df[col].unique()}")

    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def prepare_train_val_test(df_encoded: pd.DataFrame, test_size: float, random_state: int, target_col='target', mode='mlp'):

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_cols = ['MonthlyCharges', 'TotalCharges', 'charge_vs_expected', 'num_services']

    if mode == 'mlp':
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=random_state, stratify=y_temp
        )
        
        scaler = StandardScaler()
        
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols]   = scaler.transform(X_val[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    else:
        scaler = StandardScaler()
        
        X_temp[num_cols] = scaler.fit_transform(X_temp[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        return X_temp, X_test, y_temp, y_test


# ---------------------------------------------------------------------------
# Funções otimizadas para MLP (preservam mais sinal contínuo e categórico)
# ---------------------------------------------------------------------------

def mlp_create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering otimizado para MLP: mantém tenure contínuo e adiciona features derivadas."""

    # Tenure bucket (feature adicional, mas NÃO substitui tenure)
    bins = [-1, 12, 24, 48, np.inf]
    labels = ['0-12', '13-24', '25-48', '49+']
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # Avg charges per month (mantida como feature contínua)
    df['avg_charges_per_month'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    # Charge vs expected
    df['charge_vs_expected'] = df['MonthlyCharges'] - df['avg_charges_per_month']

    # Num services
    service_list = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[service_list].sum(axis=1)

    return df


def mlp_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing otimizado para MLP: preserva InternetService, SeniorCitizen,
    PaperlessBilling, Dependents e tenure contínuo."""

    df = df.copy()
    df.drop(columns=['customerID'], inplace=True)

    # Converter para binário
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'MultipleLines']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})

    # InternetService: preservar distinção DSL vs Fiber (41.9% vs 19% churn)
    df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['has_dsl'] = (df['InternetService'] == 'DSL').astype(int)
    df.drop(columns=['InternetService'], inplace=True)

    # TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']

    # Log transform em TotalCharges
    df['TotalCharges'] = np.log1p(df['TotalCharges'])

    # Feature engineering (mantém tenure contínuo)
    df = mlp_create_features(df)

    # Dropar apenas gender (baixo poder preditivo)
    df.drop(columns=['gender'], inplace=True)

    # One-Hot Encoding
    df_encoded = hot_encoding(df)

    return df_encoded


def mlp_prepare_train_val_test(df_encoded: pd.DataFrame, test_size: float,
                                random_state: int, target_col='target'):
    """Split e scaling com lista expandida de features contínuas para MLP."""

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=random_state, stratify=y_temp
    )

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'charge_vs_expected', 'num_services', 'avg_charges_per_month']

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test, y_train, y_val, y_test
