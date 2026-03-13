# Instructions for build and run container with generative models

The easiest way to work with this part of the project is to build a container on a server with an available video card.

```bash
git clone https://github.com/ITMO-NSS-team/MCPhub.git

cd SMILES_generative_models

docker build -t generative_model_mcp \
  --build-arg CSNTS_GITHUB_TOKEN=<your_github_token> \
  --build-arg CSNTS_HF_TOK=<your_hf_token> \
  --build-arg CSNTS_GEN_MOLS_MODEL_APP_PORT=8000 \
  --build-arg CSNTS_ML_MOLS_MODEL_APP_URL=http://10.32.11.22:8777 \
  --build-arg CSNTS_MOLS_GEN_MCP_PORT=8884 \
  --build-arg CSNTS_S3_ENDPOINT_URL=http://10.32.1.114:9000 \
  --build-arg CSNTS_S3_ACCESS_KEY='user' \
  --build-arg CSNTS_S3_SECRET_KEY='SECRET_KEY' \
  --build-arg CSNTS_S3_BUCKET_NAME=molecule-generative-mcp \
  -f SMILES_generative_models/Dockerfile .



```
# Running a container

The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.

Next, after you have created an image on your server (or locally), you need to run the container with the command:
```bash

docker run --name gen_models_mcp_server \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=<your_device_id> \
  -m 64G \
  --cpus="6" \
  -p 8884:8884 \
  -it --init generative_model_mcp

```
