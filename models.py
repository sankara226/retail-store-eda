import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from prophet import Prophet

def train_return_classifier(df: pd.DataFrame):
    """Executes a predictive pipeline targeting item fulfillment returns."""
    numeric_features = ['UnitPrice', 'Quantity', 'ShippingCost', 'Month', 'DayOfWeek']
    categorical_features = ['Region', 'Product', 'StoreLocation', 'Salesperson', 'CustomerType', 'HasPromotion']
    
    X = df[numeric_features + categorical_features]
    y = df['IsReturned']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    print("\n" + "="*50)
    print("      CLASSIFICATION REPORT: ITEM RETURNS")
    print("="*50)
    print(classification_report(y_test, preds))

def train_revenue_regressor(df: pd.DataFrame):
    """Executes a target leakage-free predictive regression pipeline for Revenue Estimation."""
    # Fixed leakage: Removed TotalPrice, NetUnitPrice and ComputedRevenue from features
    numeric_features = ['UnitPrice', 'Quantity', 'ShippingCost', 'Discount', 'Month', 'DayOfWeek']
    categorical_features = ['Region', 'Product', 'StoreLocation', 'Salesperson', 'CustomerType']
    
    X = df[numeric_features + categorical_features]
    y = df['ComputedRevenue']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    print("\n" + "="*50)
    print("      REGRESSION METRICS: REVENUE PREDICTION")
    print("="*50)
    print(f"Mean Absolute Error (MAE) : {mean_absolute_error(y_test, preds):.2f}")
    print(f"Coefficient of Det. (R²)  : {r2_score(y_test, preds):.3f}")

def run_prophet_forecast(df: pd.DataFrame):
    """Performs strategic timeline forecasting using an additive model architecture."""
    monthly_rev = df.groupby(pd.Grouper(key='OrderDate', freq='ME'))['ComputedRevenue'].sum().reset_index()
    monthly_rev.columns = ['ds', 'y']
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(monthly_rev)
    
    future = m.make_future_dataframe(periods=24, freq='ME')  # Focused on a highly realistic 24-month horizon
    forecast = m.predict(future)
    
    print("\n" + "="*50)
    print("      PROPHET TIME-SERIES FORECAST (TAIL)")
    print("="*50)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())