import pathlib

import networkx as nx
import plotly.express as px
import osmnx as ox
import geopandas as gpd
import shapely
import numpy as np
import plotly.graph_objects as go
import json
import os
from dash import Dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_leaflet as dl
from dash.exceptions import PreventUpdate
from haversine import haversine, Unit
import rasterio
from rasterio.sample import sample_gen
import dash_bootstrap_components as dbc
from dash_extensions.javascript import arrow_function
import re
import redis
import pickle



px.set_mapbox_access_token(open(".mapbox_token").read())
p = re.compile(r"[^a-zA-Z0-9_ -]")
r = redis.Redis()


def list_available_maps():
    d = pathlib.Path("static/edges")
    return [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in d.iterdir()]


def get_elevation_profile_of_segment(dataset: rasterio.DatasetReader, coords: list[list]):
    # coordinates are [lon, lat]
    # convert meters to feet
    coords = [[c[1], c[0]] for c in coords]
    elev = [e[0] * 3.28084 for e in sample_gen(dataset, coords)]
    d = [0.0]
    for j in range(len(coords) - 1):
        d.append(d[j] + haversine((coords[j][1], coords[j][0]), (coords[j + 1][1], coords[j + 1][0]), Unit.MILES))
    return d, elev


def coords_to_distance_vector(coords: list[list], reverse_coords: bool = True) -> list:
    if reverse_coords: # needs to be [lat, lon]
        coords = [c[::-1] for c in coords]
    return [haversine(coords[i], coords[i+1], Unit.MILES) for i in range(len(coords) - 1)]


def elevation_gain(elev):
    return sum([max(e, 0) for e in np.diff(elev)])


def load_map(place: str, graph_only: bool = False) -> (nx.MultiDiGraph, dict):
    filename = p.sub("", place).lower().replace(" ", "_")
    graph_fp = f"static/graphs/{filename}.graphml"
    edges_fp = f"static/edges/{filename}.geojson"
    nodes_fp = f"static/nodes/{filename}.geojson"
    try:
        G = ox.load_graphml(graph_fp)
        print("Graph loaded from disk")
    except FileNotFoundError:
        print("Downloading and caching graph from OSM")
        cf = '["highway"~"path|track|footway|steps|bridleway|cycleway"]'
        G = ox.graph_from_place(place, custom_filter=cf)
        ox.save_graphml(G, graph_fp)
    if graph_only:
        return G
    if os.path.isfile(edges_fp) is False or os.path.isfile(nodes_fp) is False:
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges[["name", "geometry", "length"]]
        edges["name"] = edges["name"].apply(lambda x: x[0] if type(x) is list else x)
        edges.to_file(edges_fp)
        nodes.to_file(nodes_fp)
    with open(nodes_fp, "r") as f:
        nodes = json.load(f)
    with open(edges_fp, "r") as f:
        edges = json.load(f)
    for feat in edges["features"]:
        tooltip = f"{feat['properties']['name']}, {round(feat['properties']['length'] / 1609, 1)} mi"
        feat["properties"]["tooltip"] = tooltip
    return G, nodes, edges


def plot_gdf(geo_df: gpd.GeoDataFrame):
    lats = []
    lons = []
    names = []

    for feature, name in zip(geo_df.geometry, geo_df.name):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)

    fig = go.FigureWidget(px.line_mapbox(lat=lats, lon=lons,
                         mapbox_style="outdoors", zoom=10))

    return fig


# Initialize
opts = list_available_maps()
G, _, edges = load_map(opts[0]["label"]) if len(opts) > 0 else (None, None, None)
r.set("graph", pickle.dumps(G))

polyline = dl.Polyline(id="route-line", positions=[], color="red", weight=10, fillColor="red", fillOpacity=1.0)
patterns = [dict(repeat='100',
                 arrowHead=dict(pixelSize=15, polygon=False, pathOptions=dict(color="red", stroke=True)),
                 line=dict(pixelSize=10, pathOptions=dict(color='#f00', weight=20)))]
route_decorator = dl.PolylineDecorator(id="route-arrows", positions=[], patterns=patterns)

app = Dash(prevent_initial_callbacks=True, external_stylesheets=[dbc.themes.COSMO])
app.layout = html.Div([
    html.H2("Trail Mapping"),
    html.Div([
        dbc.Label(html_for="map-select", children="Select from Available Maps"),
        dbc.Select(id="map-select", options=opts, value=opts[0]["value"] if len(opts) > 0 else None),
        dbc.Label(html_for="custom-search", children="Download a New Map"),
        dbc.Input(id="custom-search", placeholder="Shenandoah National Park, Virginia, USA"),
        dbc.Button(id="search-btn", children="Custom Search"),
        dbc.Alert(id="error-msg", children="Issue downloading map. Please try a different query in 'City, State, Country' format.", is_open=False, dismissable=True)
    ]),
    dl.Map([dl.TileLayer(),
            polyline,
            dl.GeoJSON(data=edges, zoomToBounds=True, id="trail", hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray=''))),
            # polyline,
            route_decorator
            ],
           id="map",
           style={'width': '100%',
                  'height': '60vh',
                  'margin': "auto",
                  "display": "block"}),
    dbc.Button(id="reset-btn", children="Reset"),
    # dbc.Button(id="undo-btn", children="Undo"),
    html.P(id="total-distance"),
    html.P(id="elevation-gain"),
    dcc.Graph(id="profile"),
    dcc.Store(id="route-path", data=[]),
    dcc.Store(id="map-data", data={"edges": edges})
])


@app.callback(
    Output("map-data", "data"),
    Output("trail", "data"),
    Input("map-select", "value")
)
def select_map(place):
    print(f"select_map: {place}")
    G, _, edges = load_map(place)
    map_data = {"edges": edges}
    r.set("graph", pickle.dumps(G))
    return map_data, edges


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


@app.callback(
    Output("route-path", "data"),
    Input("map", "click_lat_lng"),
    State("route-path", "data"),
    # State("map-data", "data"),
    # Input("map-select", "value")
)
def map_click(click_lat_lng, path):
    if click_lat_lng is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    # G = load_map(place, True)
    # G = nx.from_dict_of_lists(map_data["graph"])
    G = pickle.loads(r.get("graph"))
    nearest_node_id = ox.nearest_nodes(G, click_lat_lng[1], click_lat_lng[0])
    nearest_node = G.nodes[nearest_node_id]
    path = path or {"points": [], "nodes": [], "last_click_idx": 0}
    if len(path["points"]) == 0:
        path["points"].append([nearest_node["y"], nearest_node["x"]])
    else:
        pt = ox.shortest_path(G, path["nodes"][-1], nearest_node_id)
        for u, v in zip(pt[:-1], pt[1:]):
            # if there are parallel edges, select the shortest
            data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
            if "geometry" in data:
                # if geometry attribute exists, add all its coords to list
                xs, ys = data["geometry"].xy
                pts = [[ys[i], xs[i]] for i in range(len(xs))]
                path["points"] += pts
            else:
                # otherwise, the edge is a straight line from node to node
                path["points"].append([G.nodes[u]["y"], G.nodes[u]["x"]])
    path["nodes"].append(nearest_node_id)
    path["last_click_idx"] = len(path["points"])
    return path


@app.callback(
    Output("route-line", "positions"),
    Output("route-arrows", "positions"),
    Output("profile", "figure"),
    Output("elevation-gain", "children"),
    Output("total-distance", "children"),
    Input("route-path", "modified_timestamp"),
    State("route-path", "data")
)
def on_data(ts, path):
    if ts is None:
        raise PreventUpdate
    path = path or {"points": [], "nodes": [], "last_click_idx": 0}

    with rasterio.open("static/gis-data/SRTM_GL3/SRTM_GL3_srtm.vrt") as dataset:
        if len(path["points"]) > 0:
            d, elev = get_elevation_profile_of_segment(dataset, path["points"])
        else:
            d, elev = [0.0], [0.0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=elev, line={"width": 4},
                             hovertemplate='<b>Distance</b>: %{x:.2f} mi<br><b>Elevation</b>: %{y:.0f} ft'))
    fig.update_layout(title="Elevation Profile",
                      yaxis_title="Elevation (ft)",
                      xaxis_title="Distance (mi)",
                      hovermode="x")
    elev_gain = int(elevation_gain(elev))
    total_distance = round(d[-1], 1)
    return path.get("points"), path.get("points"), fig, f"Elevation Gain: {elev_gain} ft", f"Distance: {total_distance} mi"


@app.callback(
    Output("route-path", "clear_data"),
    Input("reset-btn", "n_clicks")
)
def reset(n_clicks):
    if n_clicks and n_clicks > 0:
        return True
    return False


if __name__ == '__main__':
    app.run_server(debug=False)

