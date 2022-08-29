import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, callback
import dash_leaflet as dl
from dash.exceptions import PreventUpdate
from mapping_utils import *
import dash_bootstrap_components as dbc
from shapely.geometry import LineString
from dash_extensions.javascript import arrow_function
from uuid import uuid4


dash.register_page(__name__, path_template="/map/<place>")

px.set_mapbox_access_token(open(".mapbox_token").read())


def layout(place=None):
    if place:
        global G, G2
        G, _, edges = load_map(place)
        G2 = nx.Graph(G)

        polyline = dl.Polyline(id="route-line", positions=[], color="red", weight=10, fillColor="red", fillOpacity=1.0)
        patterns = [dict(repeat='100',
                         arrowHead=dict(pixelSize=15, polygon=False, pathOptions=dict(color="red", stroke=True)),
                         line=dict(pixelSize=10, pathOptions=dict(color='#f00', weight=20)))]
        route_decorator = dl.PolylineDecorator(id="route-arrows", positions=[], patterns=patterns)

        url = "https://api.mapbox.com/styles/v1/mapbox/outdoors-v9/tiles/{z}/{x}/{y}?access_token=" + open(".mapbox_token").read()
        attribution = '&copy; <a href="https://openstreetmap.org/">OpenStreetMap</a> | &copy; <a href="https://mapbox.com/">Mapbox</a> '

        res = html.Div([
            html.Div([
                dbc.Label("Choose Mode"),
                dbc.RadioItems(
                    options=[
                        {"label": "Auto", "value": 1},
                        {"label": "Custom", "value": 2}
                    ],
                    value=1,
                    id="mode-select"
                )
            ]),
            dbc.Label("Distance Target (mi)", html_for="distance-target"),
            dbc.Input(id="distance-target", value=10, required=True, type="number", min=0),
            dbc.Label("Elevation Minimum (ft)", html_for="elevation-min"),
            dbc.Input(id="elevation-min", value=0, required=True, type="number", min=0),
            dl.Map([dl.TileLayer(url=url, attribution=attribution),
                    polyline,
                    dl.GeoJSON(data=edges, zoomToBounds=True, id="trail", hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray=''))),
                    # polyline,
                    route_decorator
                    ],
                   id="map",
                   style={'width': '100%',
                          'height': '40vh',
                          'margin': "auto",
                          "display": "block",
                          "padding-top": "10px"}),
            dbc.Button(id="reset-btn", children="Reset", className="btn btn-danger"),
            dbc.Button(id="download-btn", children="Export Route"),
            dcc.Download(id="download-route"),
            # dbc.Button(id="undo-btn", children="Undo"),
            html.Br(),
            html.P(id="total-distance"),
            html.P(id="elevation-gain"),
            dcc.Graph(id="profile"),
            dcc.Store(id="route-path", data=[]),
            dcc.Store(id="map-data", data={"edges": edges})
        ],
        style={"padding-top": "10px"})
        return res
    else:
        return html.Div("issue")


@callback(
    Output("route-path", "data"),
    Input("map", "click_lat_lng"),
    Input("distance-target", "value"),
    Input("elevation-min", "value"),
    Input("mode-select", "value"),
    State("route-path", "data")
)
def cycles(click_lat_lng, distance_target: Optional[Union[int, float]], elevation_min: Optional[Union[int, float]], mode, path):
    if click_lat_lng is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    nearest_node_id = ox.nearest_nodes(G, click_lat_lng[1], click_lat_lng[0])

    if mode == 1:
        path = {"points": []}

        with rasterio.open("static/gis-data/SRTM_GL3/SRTM_GL3_srtm.vrt") as dataset:
            pt = find_best_loop(G2, nearest_node_id, int(distance_target), tol=1.0, min_elev=int(elevation_min), dataset=dataset)
        # path["points"] = path_to_coords_list(G, pt)
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

    elif mode == 2:
        if path:
            if "nodes" not in path:
                path = {"points": [], "nodes": [], "last_click_idx": 0}
        else:
            path = {"points": [], "nodes": [], "last_click_idx": 0}

        nearest_node = G.nodes[nearest_node_id]

        if len(path["points"]) == 0:
            path["points"].append([nearest_node["y"], nearest_node["x"]])
        else:
            pt = ox.shortest_path(G, path["nodes"][-1], nearest_node_id)
            # path["points"] = path_to_coords_list(G, pt)
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


@callback(
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


@callback(
    Output("download-route", "data"),
    Input("download-btn", "n_clicks"),
    State("route-path", "data"),
    prevent_initial_call=True
)
def download_route(n_clicks, path):
    if n_clicks is None:
        raise PreventUpdate

    tmp_fp = f"cache/{str(uuid4())}.gpx"

    pts_rev = [(pt[1], pt[0]) for pt in path["points"]]
    geom = LineString(pts_rev)
    gpd.GeoDataFrame([geom], columns=["geometry"]).to_file(tmp_fp)
    with open(tmp_fp, "r") as f:
        gpx = f.read()
    return dict(content=gpx, filename="route.gpx")


@callback(
    Output("route-path", "clear_data"),
    Input("reset-btn", "n_clicks")
)
def reset(n_clicks):
    if n_clicks and n_clicks > 0:
        return True
    return False