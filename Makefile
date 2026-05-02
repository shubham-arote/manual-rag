# ─────────────────────────────────────────────────────────────────────────────
#  PDF RAG — convenience targets
#  Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

IMAGE     := pdf-rag-api
TAG       := latest
REGISTRY  ?= docker.io/$(shell whoami)

# ── Local development ─────────────────────────────────────────────────────────

.PHONY: dev
dev:                          ## Run FastAPI server locally (no Docker)
	python scripts/index_and_serve.py api \
	    --index-dir lancedb_index --out output --port 8000

.PHONY: ui
ui:                           ## Run Gradio UI locally (no Docker)
	python scripts/index_and_serve.py serve \
	    --index-dir lancedb_index --out output --port 7860

.PHONY: test
test:                         ## Run the full test suite
	uv run pytest --tb=short

.PHONY: eval
eval:                         ## Run retrieval evaluation (needs live index)
	uv run python tests/evaluation/eval_retrieval.py \
	    --index-dir lancedb_index

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: build
build:                        ## Build the serve Docker image (index must exist)
	docker build -t $(IMAGE):$(TAG) .

.PHONY: run
run:                          ## Run the serve image locally
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE):$(TAG)

.PHONY: compose-up
compose-up:                   ## Start API via docker compose
	docker compose up --build api

.PHONY: compose-down
compose-down:                 ## Stop docker compose services
	docker compose down

# ── Deployment ────────────────────────────────────────────────────────────────

.PHONY: push
push: build                   ## Build and push to Docker Hub
	docker tag $(IMAGE):$(TAG) $(REGISTRY)/$(IMAGE):$(TAG)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)
	@echo "Pushed: $(REGISTRY)/$(IMAGE):$(TAG)"

.PHONY: deploy-railway
deploy-railway:               ## Deploy to Railway (requires: railway login)
	railway up

.PHONY: deploy-fly
deploy-fly:                   ## Deploy to Fly.io (requires: fly auth login)
	fly deploy

# ── Indexing ──────────────────────────────────────────────────────────────────

.PHONY: index
index:                        ## Re-index all PDFs in ./data/  (set PDF=path for one file)
	$(if $(PDF), \
	    python scripts/index_and_serve.py index --pdf $(PDF) --out output, \
	    @echo "Usage: make index PDF=data/your_manual.pdf")

.PHONY: help
help:                         ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*##"}; {printf "  %-20s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
