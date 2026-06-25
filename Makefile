# Convenience targets. Override CAND/OUT as needed:
#   make rank CAND=./candidates.jsonl
CAND ?= ./candidates.jsonl
OUT  ?= ./submission.csv
ART  ?= ./artifacts

.PHONY: install precompute rank eval validate test demo docker-build docker-run

install:
	pip install -r requirements.txt

precompute:        ## Offline embeddings (GPU-friendly; may exceed 5 min)
	pip install -r requirements-precompute.txt
	python precompute.py --candidates $(CAND) --artifacts $(ART)

rank:              ## Produce submission.csv (CPU-only, < 5 min)
	python rank.py --candidates $(CAND) --out $(OUT) --artifacts $(ART) --validate

eval:              ## Trap-catch + silver-proxy metrics
	python -m eval.evaluate --candidates $(CAND) --submission $(OUT)

validate:          ## Run the official format validator
	python validate_submission.py $(OUT)

test:              ## Unit + smoke tests
	python -m pytest -q

demo:              ## Launch the Streamlit sandbox locally
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t redrob-ranker .

docker-run:        ## Reproduce the ranking offline in a container
	docker run --rm --network none \
	  -v "$(PWD)/$(CAND):/app/candidates.jsonl:ro" \
	  -v "$(PWD)/$(ART):/app/artifacts:ro" \
	  -v "$(PWD)/out:/app/out" \
	  redrob-ranker --candidates /app/candidates.jsonl --out /app/out/submission.csv --validate
