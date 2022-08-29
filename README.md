# trail-mapping

Creating the perfect hike with OSM and NetworkX.

## Getting Started

1. Clone the repository with `git clone https://github.com/evandiewald/trail-mapping`
2. Bulk download the full SRTM GL3 dataset (~18GB) from [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSRTM.042013.4326.1) and place it in a subfolder called `static/gis-data`
3. Initialize a conda environment from `spec-file.txt`: `conda create --name myenv --file spec-file.txt`
4. Activate the environment with `conda activate myenv`
5. Run the dash application with `python app_multipage.py`. By default, it is hosted at `http://localhost:8050`.

![image](https://user-images.githubusercontent.com/37876940/187301467-3ea2139f-28a6-4cf3-ac00-2271aca514f6.png)

