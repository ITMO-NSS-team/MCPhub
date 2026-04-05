FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG APP_DIR=/app
WORKDIR "$APP_DIR"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    pkg-config \
    python3-dev \
    python3-pip \
    git \
    wget \
    tar \
    libxrender1 \
    libxext6 \
    libsm6 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN wget --no-check-certificate https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/autodocksuite-4.2.6-x86_64Linux2.tar && \
    tar -xvf autodocksuite-4.2.6-x86_64Linux2.tar && \
    cd x86_64Linux2 && \
    cp autodock4 autogrid4 /usr/local/bin/ && \
    cd .. && rm -rf x86_64Linux2 autodocksuite-4.2.6-x86_64Linux2.tar

ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && rm miniconda.sh

ENV PATH="$CONDA_DIR/bin:$PATH"

RUN conda config --remove channels defaults || true && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

RUN conda install -y smina vina && conda clean -a -y

RUN wget --no-check-certificate \
    https://ccsb.scripps.edu/download/532/mgltools_x86_64Linux2_1.5.7.tar.gz && \
    tar -xzf mgltools_x86_64Linux2_1.5.7.tar.gz && \
    cd mgltools_x86_64Linux2_1.5.7 && \
    mkdir -p /opt/mgltools/bin && \
    ln -sf $(which python3) /opt/mgltools/bin/python && \
    bash install.sh -d /opt/mgltools && \
    cd .. && rm -rf mgltools_x86_64Linux2_1.5.7*

RUN ln -sf /opt/mgltools/bin/* /usr/local/bin/ || true && \
    find /opt/mgltools -name "prepare_ligand4.py" -exec ln -sf {} /usr/local/bin/ \; && \
    find /opt/mgltools -name "prepare_receptor4.py" -exec ln -sf {} /usr/local/bin/ \; && \
    mkdir -p /opt/conda/bin && \
    (ln -sf /opt/mgltools/bin/pythonsh /opt/conda/bin/python2.7 || \
     ln -sf $(which python3) /opt/conda/bin/python2.7 || true)

ENV MGLTOOL_HOME=/opt/mgltools
ENV PATH="/opt/mgltools/bin:$PATH"
ENV PYTHONPATH="$APP_DIR"

RUN if [ ! -f /opt/mgltools/bin/pythonsh ]; then \
        ln -sf $(which python3) /opt/mgltools/bin/pythonsh; \
    fi

ENV MGLTOOL_PYTHON=/opt/mgltools/bin/pythonsh

ENV CUDA_VISIBLE_DEVICES=0

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
