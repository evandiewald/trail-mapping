from dash import Dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go


ways_df = pd.read_csv("static/transformed/ways.csv")
# nodes_df = pd.read_csv("static/transformed/nodes.csv")
# nodes_df.columns = ["id", "longitude", "latitude"]
#
# lon = list(nodes_df[:1000]["longitude"])
# lat = list(nodes_df[:1000]["latitude"])
app = Dash(__name__)

fig = go.Figure([go.Scattermapbox(
    mode="lines+markers",
    lon = [pt[0] for pt in ways_df.iloc[i]["points_data"]],
    lat = [pt[1] for pt in ways_df.iloc[i]["points_data"]]) for i in range(100)])

fig.update_layout(
    mapbox = {
        'accesstoken': open(".mapbox_token").read(),
        'style': "outdoors"},
    showlegend = False)
fig.update_layout(clickmode='event+select')


app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.P(id="test-output")
])


@app.callback(
    Output('test-output', 'children'),
    Input('example-graph', 'clickData')
)
def update_figure(click):
    return json.dumps(click)

if __name__ == '__main__':
    app.run_server(debug=True)