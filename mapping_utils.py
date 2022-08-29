import osmnx as ox
import os
import networkx as nx
import json
import numpy as np
from rasterio.sample import sample_gen
from haversine import haversine, Unit
import geopandas as gpd
import shapely
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import re
from typing import Optional, Union


p = re.compile(r"[^a-zA-Z0-9_ -]")


def path_to_coords_list(G: nx.Graph, path: list):
    coords_list = []
    for u, v in zip(path[:-1], path[1:]):
        # if there are parallel edges, select the shortest
        data = G.get_edge_data(u, v)
        if "geometry" in data:
            # if geometry attribute exists, add all its coords to list
            xs, ys = data["geometry"].xy
            pts = [[ys[i], xs[i]] for i in range(len(xs))]
            coords_list += pts
        else:
            # otherwise, the edge is a straight line from node to node
            coords_list.append([G.nodes[u]["y"], G.nodes[u]["x"]])
    return coords_list


def path_length(G: nx.Graph, path: list):
    dist = sum([G.edges[path[i], path[i + 1]]["length"] for i in range(len(path) - 1)]) / 1609
    return dist


def find_best_loop(G: nx.Graph, root, target_dist, tol=1.0, min_elev: Optional[Union[int, float]] = None,
                   dataset: Optional[rasterio.DatasetReader] = None):
    if min_elev and not dataset:
        raise ValueError("If asking for elevation data, you must include a rasterio dataset")
    error = 1e8
    best_path = []
    for n in G.nodes():
        if nx.has_path(G, root, n):
            shortest_path = nx.shortest_path(G, root, n)
            paths = nx.all_simple_paths(G, root, n, cutoff=10)
            for path in paths:
                if path == shortest_path:
                    continue

                path_loop = path + nx.shortest_path(G, n, root, weight="length")[1:]

                path_diversity = len(set(path_loop)) / len(path_loop)

                dist = path_length(G, path_loop)
                e_new = np.abs(dist - target_dist) / path_diversity
                if e_new < error:
                    error = e_new
                    best_path = path_loop
                if error < tol:
                    if min_elev:
                        coords_list = path_to_coords_list(G, path_loop)
                        _, elev = get_elevation_profile_of_segment(dataset, coords_list)
                        elev_gain = elevation_gain(elev)
                        if elev_gain < min_elev:
                            continue
                    return best_path
    return best_path


def get_elevation_profile_of_segment(dataset: rasterio.DatasetReader, coords: list[list]):
    """
    Get the elevation profile (distance vs. altitude) of a path segment from the list of coordinates.
    Args:
        dataset: The opened rasterio dataset for the SRTM global topography data.
        coords: The path coordinates in [[lon1, lat1], [lon2, lat2], ...] format.

    Returns: The distance (in miles) and elevation (in feet) vectors.
    """
    # coordinates are [lon, lat], flip for rasterio
    coords = [[c[1], c[0]] for c in coords]
    # convert meters to feet and use rasterio.sample.sample_gen to query each point
    elev = [e[0] * 3.28084 for e in sample_gen(dataset, coords)]
    d = [0.0]
    for j in range(len(coords) - 1):
        # use haversine distance
        d.append(d[j] + haversine((coords[j][1], coords[j][0]), (coords[j + 1][1], coords[j + 1][0]), Unit.MILES))
    return d, elev


def coords_to_distance_vector(coords: list[list], reverse_coords: bool = True) -> list:
    if reverse_coords: # needs to be [lat, lon]
        coords = [c[::-1] for c in coords]
    return [haversine(coords[i], coords[i+1], Unit.MILES) for i in range(len(coords) - 1)]


def elevation_gain(elev, thresh=70):
    last_pt = elev[0]
    elev_gain = 0
    for e in elev:
        if np.abs(e - last_pt) >= thresh:
            if e - last_pt > 0:
                elev_gain += (e - last_pt)
            last_pt = e
    return elev_gain
    # return sum([max(e, 0) for e in np.diff(elev)])


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


def load_map(place: str, graph_only: bool = False) -> (nx.MultiDiGraph, dict):
    """
    Load OSM trail data for a given region. Initially check if the graph has already been cached on disk, otherwise it will be downloaded.
    Args:
        place: The geocode of the region of interest, e.g. 'Shenandoah National Park, Virginia, USA'
        graph_only: If true, return only the NetworkX graph, not the geojson.

    Returns: The dataset as a NetworkX MultiGraph, the nodes geojson, the edges geojson

    """
    filename = p.sub("", place).lower().replace(" ", "_")
    graph_fp = f"static/graphs/{filename}.graphml"
    edges_fp = f"static/edges/{filename}.geojson"
    nodes_fp = f"static/nodes/{filename}.geojson"
    try:
        G = ox.load_graphml(graph_fp)
        print("Graph loaded from disk")
    except FileNotFoundError:
        print("Downloading and caching graph from OSM")
        # custom filter to only include walkable segments - see: https://support.alltrails.com/hc/en-us/articles/360019246411-OSM-Derivative-Database-Derivation-Methodology
        cf = '["highway"~"path|track|footway|steps|bridleway|cycleway"]'
        G = ox.graph_from_place(place, custom_filter=cf)
        ox.save_graphml(G, graph_fp)
    if graph_only:
        return G
    if os.path.isfile(edges_fp) is False or os.path.isfile(nodes_fp) is False:
        # convert Graph to GeoDataFrames and save as GeoJSON
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
        # add custom tooltip property for Leaflet visualization
        tooltip = f"{feat['properties']['name']}, {round(feat['properties']['length'] / 1609, 1)} mi"
        feat["properties"]["tooltip"] = tooltip
    return G, nodes, edges