# chemical_mcp: docker compose (profile tools) и/или ASKCOS (askcos2_core).
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ASKCOS_CORE := $(ROOT)/chemical_tools_service/retrosynthesis/ASKCOSv2/askcos2_core
COMPOSE := docker compose --profile tools

# Интерпретатор для askcos-deps (должен совпадать с тем, что вызывает deploy.sh: python3 или python)
ASKCOS_PY ?= $(shell command -v python3 2>/dev/null || command -v python)
ASKCOS_REQ := $(ASKCOS_CORE)/requirements-deploy-host.txt

.PHONY: help all all-deploy compose askcos askcos-update askcos-deps

.DEFAULT_GOAL := help

help:
	@echo "Цели:"
	@echo "  make all            — compose up + ASKCOS make update (без seed-db в Mongo)"
	@echo "  make all-deploy     — compose up + ASKCOS make deploy (первая установка, полный seed)"
	@echo "  make compose        — только docker compose (profile tools) up --build -d"
	@echo "  make askcos         — только make deploy в askcos2_core (с seed-db)"
	@echo "  make askcos-update  — только make update (образы/скрипты + старт, без seed-db)"
	@echo ""

all: compose askcos-update
	@echo "Готово: compose (profile tools) + ASKCOS update (без повторного seed Mongo)."
	@echo "Прокси retrosynthesis: http://localhost:8001"

all-deploy: compose askcos
	@echo "Готово: compose + ASKCOS deploy (включая seed-db при необходимости)."
	@echo "Прокси retrosynthesis: http://localhost:8001"

compose:
	@test -d "$(ASKCOS_CORE)" || (echo "Нет каталога ASKCOS: $(ASKCOS_CORE)" >&2; exit 1)
	cd "$(ROOT)" && $(COMPOSE) up --build -d
	@echo "Готово: только docker compose (profile tools)."

askcos-deps:
	@test -n "$(ASKCOS_PY)" || (echo "Нужен python3 или python в PATH (для pre_deploy.py)" >&2; exit 1)
	@$(ASKCOS_PY) -c "import yaml" 2>/dev/null || $(ASKCOS_PY) -m pip install pyyaml

askcos: askcos-deps
	@test -d "$(ASKCOS_CORE)" || (echo "Нет каталога ASKCOS: $(ASKCOS_CORE)" >&2; exit 1)
	cd "$(ASKCOS_CORE)" && $(MAKE) deploy
	@echo "Готово: только ASKCOS (make deploy)."

askcos-update: askcos-deps
	@test -d "$(ASKCOS_CORE)" || (echo "Нет каталога ASKCOS: $(ASKCOS_CORE)" >&2; exit 1)
	cd "$(ASKCOS_CORE)" && $(MAKE) update
	@echo "Готово: ASKCOS make update (без seed-db)."
