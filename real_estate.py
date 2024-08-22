import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px  # Import Plotly Express

# Load the data
try:
    data = pd.read_csv(r"C:\Users\kyash\OneDrive\Desktop\Indian_Real_Estate_Clean_Data.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Print the columns for debugging
print("Columns in the dataset:", data.columns)

# Preprocess the data
# Fill missing values in numerical columns with their mean
numeric_columns = ['Total_Area(SQFT)', 'Total_Rooms', 'BHK', 'Price_per_SQFT']
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Define scaled columns
scaled_columns = ['Total_Area(SQFT)', 'Total_Rooms', 'BHK']
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Rename columns to be more intuitive for modeling
data.rename(columns={
    'Total_Area(SQFT)': 'size',
    'Total_Rooms': 'num_bedrooms',
    'BHK': 'num_bhk'
}, inplace=True)

# Split the data into training and testing sets
X = data[['size', 'num_bedrooms', 'num_bhk']]  # features
y = data['Price']  # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
try:
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
except Exception as e:
    print(f"Error training linear regression model: {e}")
    raise

# Create and train a decision tree regression model
try:
    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(X_train, y_train)
except Exception as e:
    print(f"Error training decision tree regression model: {e}")
    raise

# Create and train a random forest regression model
try:
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
except Exception as e:
    print(f"Error training random forest regression model: {e}")
    raise

# Make predictions on the testing data
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Evaluate the models using mean squared error and R-squared
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Create a dashboard with different visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('India Real Estate Price Prediction'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Visualization', value='tab-1'),
        dcc.Tab(label='Model Performance', value='tab-2'),
        dcc.Tab(label='Prediction', value='tab-3')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H2('Data Visualization'),
            dcc.Graph(figure=px.scatter(data, x='size', y='Price', title='Size vs Price')),
            dcc.Graph(figure=px.scatter(data, x='num_bedrooms', y='Price', title='Number of Bedrooms vs Price')),
            dcc.Graph(figure=px.scatter(data, x='num_bhk', y='Price', title='BHK vs Price'))
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H2('Model Performance'),
            html.P(f'Linear Regression: MSE={mse_lr:.2f}, R2={r2_lr:.2f}'),
            html.P(f'Decision Tree Regression: MSE={mse_dt:.2f}, R2={r2_dt:.2f}'),
            html.P(f'Random Forest Regression: MSE={mse_rf:.2f}, R2={r2_rf:.2f}')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H2('Prediction'),
            html.P('Please select a model to make a prediction:')
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
