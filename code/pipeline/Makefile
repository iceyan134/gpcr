.PHONY: build up down logs shell cli test

# ── Docker build & run ────────────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d esmfold2-local

up-api:
	docker compose --profile api up -d esmfold2-api

up-batch:
	docker compose --profile batch up -d redis esmfold2-worker

down:
	docker compose --profile all down

logs:
	docker compose logs -f esmfold2-local

shell:
	docker compose exec esmfold2-local bash

# ── Interactive CLI ───────────────────────────────────────────────────
cli:
	docker compose run --rm esmfold2-local cli $(ARGS)

# Example: make cli-protein SEQ="MKTAYIAKQRQISFVKSHFSR"
cli-protein:
	docker compose run --rm esmfold2-local cli --protein "$(SEQ)"

# Example: make cli-complex PROT="MKTAYIAKQR" LIG="SAH"
cli-complex:
	docker compose run --rm esmfold2-local cli --protein "$(PROT)" --ligand "$(LIG)"

# ── API test ──────────────────────────────────────────────────────────
test-health:
	curl -s http://localhost:8000/health | python -m json.tool

test-fold:
	curl -s -X POST http://localhost:8000/fold \
		-H "Content-Type: application/json" \
		-d '{"name":"test","chains":[{"id":"A","sequence":"MKTAYIAKQRQISFVKSHFSR","type":"protein"}],"config":{"num_loops":1,"num_sampling_steps":8}}' \
		| python -m json.tool

# ── Clean ─────────────────────────────────────────────────────────────
clean:
	rm -rf output/*
	rm -rf cache/huggingface/*
