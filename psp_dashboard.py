import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go

df = pd.read_csv('psp_data.csv')

def trim_dataset(df):
    df = df[['Planting Year', 'Block', 'Plot Number', 'Tree Number', 'Height', 'DBH (cm)']]

    return df

def remove_dead_trees(df):
    df = df[~((df['Height'] == 'dead'))]

    return df

def transform_data(df):
    df['Height'] = df['Height'].astype(float)
    df['DBH (cm)'] = df['DBH (cm)'].astype(float)

    return df

def compute_volume(df):
    df['volume'] = 0.000039 * df['DBH (cm)']**2 * df['Height']

    return df



df = trim_dataset(df)
df = remove_dead_trees(df)
df = transform_data(df)
df = compute_volume(df)

st.title("PSP Scatter Plot Dashboard with Regression Line")

# Sidebar controls
st.sidebar.header('Plot Settings')
x_var= st.sidebar.selectbox("Select X-axis variable", ['DBH (cm)', 'Height', 'volume'], index=0)

y_var = st.sidebar.selectbox("Select Y-axis variable", ['DBH (cm)', 'Height', 'volume'], index=2)

years = sorted(df['Planting Year'].dropna().astype(str).unique())
year_filter = st.sidebar.selectbox('Filter by Planting Year', options=['All'] + list(years))


# filter data
filtered_df = df.copy()
if year_filter != "All":
    filtered_df = filtered_df[filtered_df['Planting Year'] == year_filter]

filtered_df = filtered_df[[x_var, y_var]].dropna()

# Plot
st.subheader(f"Scatter Plot: {x_var} vs {y_var}")
fig = go.Figure()

# Scatter points
fig.add_trace(go.Scatter(
        x=filtered_df[x_var],
        y=filtered_df[y_var],
        mode='markers',
        name='Data Points',
        marker=dict(color='blue', opacity=0.7)
    ))

# Regression line
if len(filtered_df) > 1:
    X = filtered_df[[x_var]].values
    y = filtered_df[y_var].values
    model = LinearRegression().fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    fig.add_trace(go.Scatter(
            x=x_range.flatten(),
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        ))

# Customize layout
fig.update_layout(
    xaxis_title=x_var,
    yaxis_title=y_var,
    margin=dict(l=40, r=40, t=40, b=40),
    height=500
)

st.plotly_chart(fig, use_container_width=True)



