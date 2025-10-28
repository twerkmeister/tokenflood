# Tokenflood

Tokenflood is a load testing tool for instruction-tuned LLMs that allows you to 
run arbitrary load profiles without needing specific prompt and response data.
**Define desired prompt lengths, prefix lengths, output lengths, and request rates, 
and tokenflood simulates this workload for you.** 

Tokenflood makes it easy to explore the effects of using different providers, 
deploying different hardware, using quantization, or altering prompt and output lengths.

Tokenflood uses [litellm](https://www.litellm.ai/) under the hood and supports 
[all providers that litellm covers](https://docs.litellm.ai/docs/providers).

> [!CAUTION]
> Tokenflood can generate high costs if configured poorly and used with pay-per- 
> token services. Make sure you only test workloads that are within a reasonable budget.

## Common Usage Scenarios

1. Load testing self-hosted LLMs
2. Assessing the effects of hardware, quantization, and prompt optimizations
3. Assessing the intraday latency variations of hosted LLM providers


## Installation

```bash
pip install tokenflood
```

## Quick Start

For the quick start, make sure that vllm is installed, and you serve a small model:
```bash
pip install vllm
vllm serve HuggingFaceTB/SmolLM-135M-Instruct
```

Afterward, create the basic config files and do a first run:
```bash
# This creates config files for a tiny first run: run.yml and endpoint.yml
tokenflood init
# Afterwards you can inspect those files and run them
tokenflood run run.yml endpoint.yml
```

Finally, in the `results` folder you should find your run folder
containing a graph visualizing the latency quantiles across the difference request 
rates, a summary file, the run- and endpoint configs and an error log in case something
went wrong.

## Endpoint Specs

With the endpoint spec file you can determine the target of the load test. 
Tokenflood uses [litellm](https://www.litellm.ai/) under the hood and supports 
[all providers that litellm covers](https://docs.litellm.ai/docs/providers).

Here you see the example endpoint spec file from the quick start: 
```yaml
provider: hosted_vllm
model: HuggingFaceTB/SmolLM-135M-Instruct
base_url: http://127.0.0.1:8000/v1
api_key_env_var: null
deployment: null
extra_headers: {}
```
Explanation of the parameters
* `provider`: is the provider parameter used by litellm and is used to determine how to exactly interact with the endpoint as different providers offer different APIs.
* `model`: the specific model to use at the given endpoint.
* `base_url`: important if you are self-hosting or using an endpoint in a specific region of a provider.
* `api_key_env_var`: The name of the environment variable to use as the API key. If you specify it, it allows you to manage multiple API keys for the same provider for different regions without changing env files: such as `AZURE_KEY_FRANKFURT` and `AZURE_KEY_LONDON`.
* `deployment`: Required for some providers such as azure.
* `extra_headers`: Can be useful for certain providers to select models.

Tokenflood passes all these parameters right through to litellm's completion call. So it's 
best to have a look at [the official documentation of the completion call](https://docs.litellm.ai/docs/completion/input) 

### Endpoint Examples

* Self-hosted VLLM
* Openai
* Bedrock
* Azure
* Gemini
* 
