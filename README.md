# geojson_service


## Setup
docker build -f Dockerfile -t geojson-service:v1 .
docker run -v /datadrive/:/datadrive --name geojson-test-v1 -it geojson-service:v1 bash

python libs/geojson_2_babymaps.py /datadrive/geojson/D8-aoi-water-Final.geojson /datadrive/geojson/D8-aoi-water-Final-outputs/ /datadrive/D8_2024-02-26 --format .jpg

