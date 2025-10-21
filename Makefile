
test:
	pytest --cov=tokenflood tests

ci-vllm:
	vllm serve HuggingFaceTB/SmolLM-135M-Instruct

lint:
	poetry run ruff check tokenflood tests --fix
	poetry run ruff format tokenflood tests
	poetry run mypy tokenflood tests
