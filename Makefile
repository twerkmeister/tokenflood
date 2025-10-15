
test:
	pytest tests

ci-vllm:
	vllm serve HuggingFaceTB/SmolLM-135M-Instruct
