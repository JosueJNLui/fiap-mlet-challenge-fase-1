from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocessing(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:

    # Remover customerID (identificador, não preditivo)
    df_model = df.drop(columns=['customerID'])

    # Converter TotalCharges para numérico (11 valores em branco → NaN)
    df_model['TotalCharges'] = pd.to_numeric(df_model['TotalCharges'], errors='coerce')

    print(f"Missing values em TotalCharges: {df_model['TotalCharges'].isna().sum()}")
    print(f"Registros afetados:\n{df_model[df_model['TotalCharges'].isna()][['tenure', 'MonthlyCharges', 'TotalCharges']]}\n")

    # --- Train/Test Split (ANTES de imputar para evitar data leakage) ---
    X = df_model.drop(columns=['target'])
    y = df_model['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Criar cópias explícitas para evitar SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    print(f"Shape treino: {X_train.shape}")
    print(f"Shape teste:  {X_test.shape}")

    # --- Imputação: fit no treino, transform em ambos ---
    imputer = SimpleImputer(strategy='median')

    # Aplicar apenas na coluna TotalCharges
    X_train['TotalCharges'] = imputer.fit_transform(X_train[['TotalCharges']])
    X_test['TotalCharges'] = imputer.transform(X_test[['TotalCharges']])

    print(f"\nMissing values após imputação (treino): {X_train['TotalCharges'].isna().sum()}")
    print(f"Missing values após imputação (teste):  {X_test['TotalCharges'].isna().sum()}")
    print(f"Mediana usada para imputação: {imputer.statistics_[0]:.2f}")

    return X_train, X_test, y_train, y_test

def hot_encoding(df: pd.DataFrame) -> pd.DataFrame:

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Colunas categóricas ({len(categorical_cols)}): {categorical_cols}\n")

    # Mostrar cardinalidade de cada coluna categórica
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} valores únicos → {df[col].unique()}")

    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
