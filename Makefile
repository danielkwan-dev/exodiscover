.PHONY: install ingest train eval test lint serve web docker
install:
	pip install -e ".[dev,api]"
	cd web && npm install

ingest:
	exo ingest

train:
	exo train

eval:
	exo eval

test:
	pytest
	cd web && npm run test

lint:
	ruff check .
	mypy
	cd web && npm run lint

serve:
	flask --app api.wsgi run --port 8000 --reload

web:
	cd web && npm run dev

docker:
	docker compose up --build
