import pathlib
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from mapping_utils import load_map



def list_available_maps():
    d = pathlib.Path("static/edges")
    return [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in d.iterdir()]


# Initialize
opts = list_available_maps()


app = Dash(prevent_initial_callbacks=True, external_stylesheets=[dbc.themes.COSMO], use_pages=True)
app.layout = html.Div([
    html.H2("Trail Mapping"),
    dbc.Accordion([
        dbc.AccordionItem([
            html.Div([
                dbc.Label(html_for="map-select", children="Select from Available Maps"),
                dbc.Select(id="map-select", options=opts, value=opts[0]["value"] if len(opts) > 0 else None),
                dbc.Label(html_for="custom-search", children="Download a New Map"),
                dbc.Input(id="custom-search", placeholder="Shenandoah National Park, Virginia, USA"),
                dbc.Button(id="search-btn", children="Custom Search"),
                dbc.Alert(id="error-msg", children="Issue downloading map. Please try a different query in 'City, State, Country' format.", is_open=False, dismissable=True),
                dcc.Link('Dashboard', id="map-link", href="/map/"+opts[0]["value"] if len(opts) > 0 else "#", className="btn btn-dark")])
        ], title="Select Map")
    ]),
    dash.page_container
],
style={"padding-left": "10%", "padding-right": "10%", "padding-top": "5%"})


@app.callback(
    Output("map-link", "href"),
    Input("map-select", "value")
)
def select_map(place):
    print(f"select_map: {place}")
    print(dash.page_registry['pages.map']['relative_path'])
    url = f"/map/{place}"
    return url


@app.callback(
    Output("map-select", "options"),
    Output("error-msg", "is_open"),
    Input("search-btn", "n_clicks"),
    Input("custom-search", "value")
)
def download_map(n_clicks, place):
    if not n_clicks:
        raise PreventUpdate
    if n_clicks > 0:
        try:
            _, _ = load_map(place)
            is_open = False
        except ValueError:
            is_open = True
        opts = list_available_maps()
        return opts, is_open


if __name__ == '__main__':
    app.run_server(debug=False)

