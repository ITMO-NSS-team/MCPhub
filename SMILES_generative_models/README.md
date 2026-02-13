# Instructions for build and run container with generative models

The easiest way to work with this part of the project is to build a container on a server with an available video card.

```
git clone https://github.com/ITMO-NSS-team/MCPhub.git

cd SMILES_generative_models

docker build -t generative_model_mcp --build-arg GITHUB_TOKEN=<your token> --build-arg HF_TOK=<your token> --build-arg GEN_APP_PORT=99 --build-arg ML_MODEL_URL=10.32.11.22 --build-arg MCP_PORT=8883 .



```
# Running a container

The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.

Next, after you have created an image on your server (or locally), you need to run the container with the command:
```

docker run --name gleb_mcp --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<your device ID> -m  64G --cpus="6" -p 8883:8883 -it --init generative_model_mcp

```