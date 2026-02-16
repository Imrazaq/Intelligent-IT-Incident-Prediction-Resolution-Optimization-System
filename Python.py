mkdir IT-Incident-Intelligence-System
cd IT-Incident-Intelligence-System
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install pandas numpy scikit-learn xgboost matplotlib seaborn nltk shap streamlit

import pandas as pd

df = pd.read_csv("tickets.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df['created_at'] = pd.to_datetime(df['created_at'])
df['resolved_at'] = pd.to_datetime(df['resolved_at'])

df['resolution_time_hours'] = (
    df['resolved_at'] - df['created_at']
).dt.total_seconds() / 3600

df['sla_breach'] = df['resolution_time_hours'].apply(
    lambda x: 1 if x > 24 else 0
)

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df['description'])


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X_text, df['sla_breach'], test_size=0.2, random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from xgboost import XGBRegressor

reg_model = XGBRegressor()
reg_model.fit(X_train, df.loc[X_train.indices, 'resolution_time_hours'])

pred_time = reg_model.predict(X_test)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(
    df.loc[X_test.indices, 'resolution_time_hours'],
    pred_time
))

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)

df['date'] = df['created_at'].dt.date
daily_incidents = df.groupby('date').size()


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(daily_incidents, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=7)
print(forecast)

