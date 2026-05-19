FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Установка временной зоны
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Build-time arg used ONLY for the build-time `huggingface-cli download` below
# (only needed when pulling a private model). Defaults to empty so it does not
# end up baked into image layers. Pass via `docker build --build-arg HF_TOK=...`
# if the model becomes private. Runtime HF_TOKEN is loaded from `.env`.
ARG HF_TOK=""

# Установка базовых зависимостей
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential curl wget git cmake vim pkg-config unzip \
    libgtk2.0-dev imagemagick graphviz && \
    mkdir /projects && \
    rm -rf /var/lib/apt/lists/*

# Установка Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    export PATH=~/miniconda3/miniconda/bin:$PATH && \
    ~/miniconda3/bin/conda init bash && \
    ~/miniconda3/bin/conda init zsh


ENV PATH ~/miniconda3/miniconda/bin:$PATH
ENV LB_LIBRARY_PATH="~/miniconda3/miniconda/lib:\$LB_LIBRARY_PATH"
SHELL ["/bin/bash", "-ic"]



# Then proceed with updates
RUN conda config --set plugins.auto_accept_tos yes && \
    conda update -n base conda -y && \
    conda update --all -y


RUN conda create -n Mol_gen_env python=3.8 && \
    conda deactivate && \
    conda activate Mol_gen_env && \ 
    conda install conda-libmamba-solver && \
    conda install -c "conda-forge/label/cf202003" openbabel --solver libmamba
    #pip install -r requirements.txts
    #pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116


# RUN cd /projects &&\
#     git clone https://${GITHUB_TOKEN}@github.com/ITMO-NSS-team/MCPhub.git
COPY . /projects/generative_models
WORKDIR /projects/generative_models

RUN ~/miniconda3/envs/Mol_gen_env/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN ~/miniconda3/envs/Mol_gen_env/bin/pip install -r requirements.txt

RUN conda create -n MCP_env python=3.13 && \
    ~/miniconda3/envs/MCP_env/bin/pip install -r requirements.mcp.txt

RUN HF_TOKEN="$HF_TOK" ~/miniconda3/envs/Mol_gen_env/bin/huggingface-cli download \
    SoloWayG/Molecule_transformer --local-dir autotrain/

# Non-secret runtime path setup. All secrets and runtime config (S3 creds,
# HF_TOKEN, ports, ML_MOLS_MODEL_APP_URL, ...) come from a `.env` file at
# container start time. See `.env.example` for the full list and
# `docker-compose.yml` (env_file: .env) for wiring. Python-dotenv
# (`load_dotenv()` in main_mcp.py / main_api.py) reads it from the WORKDIR
# (/projects/generative_models/.env). Alternatively use
# `docker run --env-file .env`.
ENV LD_LIBRARY_PATH="/opt/conda/envs/Mol_gen_env/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/projects/generative_models:$PYTHONPATH"
ENV PYTHONPATH="/projects/CoScientist/:$PYTHONPATH"
ENV PYTHONPATH="/projects/:$PYTHONPATH"

# Запуск
# Strip CRLF defensively: if someone clones the repo on Windows with
# core.autocrlf=true, api.sh ends up with \r\n line endings and bash on
# Linux reads each trailing \r as part of the last token (cd target path,
# conda env name, etc.) — every command silently fails. `sed` makes the
# shipped image immune regardless of the developer's local git config.
RUN sed -i 's/\r$//' /projects/generative_models/api.sh \
    && chmod +x /projects/generative_models/api.sh
ENTRYPOINT ["/bin/bash", "/projects/generative_models/api.sh"]

