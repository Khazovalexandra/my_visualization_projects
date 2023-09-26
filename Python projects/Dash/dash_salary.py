import dash
from dash import html
from dash import dcc
import pandas as pd
import plotly.express as px

df = pd.read_csv(r'C:\Users\admin\Documents\GitHub\my_visualization_projects\Power BI\Data Science Salary 2021 to 2023.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ]),

    ])

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor':''},
                      children=[
    html.H1(children='Data Science Salary 2021 to 2023'),
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Applied Scientist', 'value': 'AS'},
            {'label': u'Data Quality Analyst', 'value': 'DQA'},
            {'label': 'Compliance Data Analyst', 'value': 'CDA'},
            {'label': 'Machine Learning Engineer', 'value': 'MLE'},            
            {'label': 'Research Scientist', 'value': 'RS'},            
            {'label': 'Data Engineer', 'value': 'DE'},            
            {'label': 'Data Analyst', 'value': 'DA'},            
        ],
        value='DA'
    ),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df['work_year'] , 'y': df['salary'] , 'type': 'bar', },
            ],
            'layout': {
                'title': 'Visualization of Salary by Time', 
                'colorway': ['#17B897'],
            },
        }
    ),
    html.H3(children='Table'),
    generate_table(df)], 
)

if __name__ == '__main__':
    app.run_server(debug=True)