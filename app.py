import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

px.set_mapbox_access_token("pk.eyJ1IjoiZXZhbmRpZXdhbGQiLCJhIjoiY2t5ZDNxZGduMDJnYTJzcDZjbTc0a2k4ZSJ9.40GKtHZlbadpxGMRJD6JBQ")


df = pd.read_csv("static/geojson/oh-trails.csv")

with open("static/geojson/oh-trails.json", "r") as f:
    geo = json.load(f)

for g in geo["features"]:
    g["id"] = g["properties"]["ObjectID"]

paths = []
for g in geo["features"][:1000]:
    if g["geometry"]["type"] == "MultiLineString":
        for line in g["geometry"]["coordinates"]:
            paths.append({"lat": np.array(line)[:,1],
                          "lon": np.array(line)[:,0]})
    else:
        paths.append({"lat": np.array(g["geometry"]["coordinates"])[:, 1],
                      "lon": np.array(g["geometry"]["coordinates"])[:, 0]})

fig = go.Figure(
    data=[
        go.Scattermapbox(
            lat=p["lat"],
            lon=p["lon"],
            mode="lines",
            line=dict(width=8)
        )
        for p in paths
    ]
)

fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    mapbox=go.layout.Mapbox(
        style="outdoors",
        zoom=6,
    ),
    mapbox_accesstoken="pk.eyJ1IjoiZXZhbmRpZXdhbGQiLCJhIjoiY2t5ZDNxZGduMDJnYTJzcDZjbTc0a2k4ZSJ9.40GKtHZlbadpxGMRJD6JBQ"
)
fig.update_layout(showlegend=False)
fig.show()

