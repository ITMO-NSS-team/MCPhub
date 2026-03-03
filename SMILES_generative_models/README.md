# Instructions for build and run container with generative models

The easiest way to work with this part of the project is to build a container on a server with an available video card.

```
git clone https://github.com/ITMO-NSS-team/MCPhub.git

cd SMILES_generative_models

docker build -t generative_model_mcp --build-arg GITHUB_TOKEN=<your token> --build-arg HF_TOK=<your token> --build-arg GEN_MOLS_MODEL_APP_PORT=<your desired (unused) port for gen models> --build-arg ML_MOLS_MODEL_APP_URL=<your url (IP with PORT) where your deploed ML modules for properties predictions> --build-arg MOLS_GEN_MCP_PORT=<your desired (unused) port for gen MCP server> .



```
# Running a container

The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.

Next, after you have created an image on your server (or locally), you need to run the container with the command:
```

docker run --name gen_models_mcp_server --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<your device ID> -m  64G --cpus="6" -p 8883:8883 -it --init generative_model_mcp

```
