
test:
	pytest tests

ci-vllm:
	vllm serve HuggingFaceTB/SmolLM-135M-Instruct

lint:
	poetry run mypy tokenflood tests
	poetry run ruff check tokenflood tests --fix
	poetry run ruff format tokenflood tests
