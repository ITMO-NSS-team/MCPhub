# UV environments for this repo

Ниже описаны два отдельных окружения Python, которые нужны для этого репозитория:

- `molgen` (Python 3.8) — зависимости из `requirements.txt` (соответствует окружению из `Dockerfile`).
- `mcp` (Python 3.13) — зависимости из `requirements.mcp.txt`.

В репозитории уже есть `pyproject.toml` с `requires-python = ">=3.12"`. Для этих двух окружений он не используется.

## Предварительные условия

- Установлен `uv`.
- Доступны интерпретаторы Python 3.8 и 3.13.  
  Если `uv` не находит нужную версию, установите ее любым удобным способом (системный Python, pyenv, uv toolchain и т.д.).

## Окружение 1: molgen (Python 3.8)

1. Создайте venv:
   ```powershell
   uv venv .venv-molgen --python 3.8
   ```
2. Активируйте:
   ```powershell
   .\.venv-molgen\Scripts\activate
   ```
   На Linux/macOS:
   ```bash
   source .venv-molgen/bin/activate
   ```
3. Установите зависимости:
   ```powershell
   uv pip install -r requirements.txt
   ```

Примечание по `torch`: в `Dockerfile` он ставится отдельно (`torch==1.12.1+cu116`, `torchvision==0.13.1+cu116`).  
Если нужна GPU-сборка, установите подходящую версию `torch` под ваш CUDA/CPU перед запуском моделей.

## Окружение 2: mcp (Python 3.13)

1. Создайте venv:
   ```powershell
   uv venv .venv-mcp --python 3.13
   ```
2. Активируйте:
   ```powershell
   .\.venv-mcp\Scripts\activate
   ```
   На Linux/macOS:
   ```bash
   source .venv-mcp/bin/activate
   ```
3. Установите зависимости:
   ```powershell
   uv pip install -r requirements.mcp.txt
   ```

## Шаблоны pyproject.toml

Добавлены два файла-шаблона:

- `pyproject.molgen.toml` — зависимости из `requirements.txt`, Python 3.8.
- `pyproject.mcp.toml` — зависимости из `requirements.mcp.txt`, Python 3.13.

Используйте их как основу, если хотите вести зависимости через `pyproject.toml`:

1. Скопируйте нужный файл и назовите его `pyproject.toml` (желательно в отдельной рабочей папке или временно, чтобы не мешать текущему `pyproject.toml`).
2. Установите зависимости привычной для вас командой `uv` для проектов на `pyproject.toml` (например, `uv sync` или `uv pip install -e .`).
