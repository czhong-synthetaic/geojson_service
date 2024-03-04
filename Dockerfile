# Ubuntu 20.04
# FROM python:3.8.18-slim-bullseye

# Ubuntu 22.04
FROM python:3.8.18-slim-bookworm

WORKDIR /root/
COPY . .

RUN echo "Installing system libraries..."
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential apt-utils wget curl tar git && \
    apt-get install -y gdal-bin gdal-data libgdal-dev python3-gdal && \
    apt-get autoclean -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN echo "Downloading azcopy..."
RUN wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && \
    tar -xf azcopy_v10.tar.gz --strip-components=1 && \
    chmod -R 757 azcopy && \
    cp azcopy /usr/bin/azcopy

RUN echo "Creating python environment..."
RUN python3.8 -m venv /root/venv_py38
ENV PATH=/root/venv_py38:$PATH
RUN pip install -U pip && \
    pip install -r requirements.txt





### Download source imagery
# ENV GLOBAL_PROD_SAS_KEY=${GLOBAL_PROD_SAS_KEY:-?sv=2022-11-02&ss=bfqt&srt=sco&sp=rltf&se=2025-02-22T00:56:36Z&st=2024-02-21T16:56:36Z&spr=https&sig=8eiBI1IhuRXam9TsFNI2ucSCGndaHM8w7CzOdEh4fzI%3D}
# ENV GUARD_TIF_STORAGE_ACCOUNT=${GUARD_TIF_STORAGE_ACCOUNT:-https://guardstscus.blob.core.windows.net/planet-dls}
# ENV GUARD_TIF_SAS_KEY=${GUARD_TIF_SAS_KEY:-?sp=rl&st=2024-02-14T00:45:32Z&se=2025-02-14T08:45:32Z&spr=https&sv=2022-11-02&sr=c&sig=HJdHWPNMzRxjwiaEF1vsbnFQDeU1cHImFgOW%2Bu46G0M%3D}
# ENV GUARD_DATAFRAME_STORAGE_ACCOUNT=${GUARD_DATAFRAME_STORAGE_ACCOUNT:-https://guardstscus.blob.core.windows.net/reference-dataframes}
# ENV GUARD_DATAFRAME_SAS_KEY=${GUARD_DATAFRAME_SAS_KEY:-?sp=rl&st=2024-02-14T02:43:18Z&se=2025-02-14T10:43:18Z&spr=https&sv=2022-11-02&sr=c&sig=xHidffOdzqBjSlEyo6Qbva7RECRe6wOayMnKT%2BivX7A%3D}
# ENV FETCH_DATASOURCE=${FETCH_DATASOURCE}
# ENV OUTPUT_GEOTIFF_FOLDER=${OUTPUT_GEOTIFF_FOLDER}
# RUN python libs/fetch_geojson_tif.py ${FETCH_DATASOURCE} ${OUTPUT_GEOTIFF_FOLDER}

### Create baby artifacts
# ARG FEATURES_GEOJSON
# ENV FEATURES_GEOJSON=${FEATURES_GEOJSON}

# ARG OUTPUT_DIR
# ENV OUTPUT_DIR=${OUTPUT_DIR}

# ARG GEOTIFF_FOLDER
# ENV GEOTIFF_FOLDER=${GEOTIFF_FOLDER}

# ARG BABY_FORMAT
# ENV BABY_FORMAT=${BABY_FORMAT}

# CMD python libs/geojson_2_babymaps.py ${FEATURES_GEOJSON} ${OUTPUT_DIR} ${GEOTIFF_FOLDER} --format ${BABY_FORMAT}
