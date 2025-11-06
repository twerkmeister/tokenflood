# Tokenflood

Tokenflood is a load testing tool for instruction-tuned LLMs that allows you to 
run arbitrary load profiles without needing specific prompt and response data.
**Define desired prompt lengths, prefix lengths, output lengths, and request rates, 
and tokenflood simulates this workload for you.** 

Tokenflood makes it easy to explore how latency changes when using different providers, 
hardware, quantizations, or prompt and output lengths.

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
Explanation of the parameters:
* `provider`: is the provider parameter used by litellm and is used to determine how to exactly interact with the endpoint as different providers have different APIs.
* `model`: the specific model to use at the given endpoint.
* `base_url`: important if you are self-hosting or using an endpoint in a specific region of a provider.
* `api_key_env_var`: The name of the environment variable to use as the API key. If you specify it, it allows you to manage multiple API keys for the same provider for different regions without changing env files: such as `AZURE_KEY_FRANKFURT` and `AZURE_KEY_LONDON`.
* `deployment`: Required for some providers such as azure.
* `extra_headers`: Can be useful for certain providers to select models.

Tokenflood passes all these parameters right through to litellm's completion call. 
To get a better understanding, it's best to have a look at [the official documentation of the completion call](https://docs.litellm.ai/docs/completion/input). 

### Endpoint Examples

**Self-hosted VLLM**

```yaml
provider: hosted_vllm
model: meta-llama/Llama-3.1-8B-Instruct
base_url: http://127.0.0.1:8000/v1
```

* Openai

```yaml
provider: openai
model: gpt-4o
```

* Bedrock
```yaml
provider: ?
model: gpt-4o
```

* AWS Sagemaker Inference Endpoints

* Azure

```yaml
provider: azure
deployment: gpt-4o
model: gpt-4o
api_version: 2024-06-01
api_base: https://my-azure-url.openai.azure.com/
```

* Gemini
* Anthropic


## Run Suites

With a run suite you define the specific test you want to run. Each test can have multiple
phases with a different number of requests per second. All phases share the same length in 
seconds and the type of load that is being send.

Here is the run suite that is being created for you upon calling `tokenflood init`:
```yaml
name: ripple
requests_per_second_rates:  # Defines the phases with the different request rates
- 1.0
- 2.0
test_length_in_seconds: 10  # each phase is 10 seconds long
load_types:                 # we have two load types with equal weight
- prompt_length: 512        # prompt length in tokens
  prefix_length: 128        # prompt prefix length in tokens
  output_length: 64         # output length in tokens
  weight: 1.0               # sampling weight for this load type
- prompt_length: 640
  prefix_length: 568
  output_length: 12
  weight: 1.0
percentiles:                # the latency percentiles to report
- 50
- 90
- 99
```

## Heuristic Load Testing

Tokenflood does not need to use specific prompt data to run tests. This allows for swift 
testing of alternative configurations and loads. Changing the token counts in the 
load types is a matter of seconds as opposed to having to adjust implementations and 
reobserving prompts of the altered system.  

This capability comes with a risk. When constructing the prompt, the heuristic may fail
to perfectly achieve the desired token counts for specific models or loads. If that 
happens, tokenflood will output a warning whenever the desired token counts diverge
more than 5% from the desired lengths. 

### A note on prefix lengths

You can specify the prefix length, however, whether the prefix is used will depend on the 
specific endpoint and its configuration. Some providers, like openai, will only start to
use prefix caching once your total prompt length exceeds 1024 tokens. Additionally,
it seems litellm does not always record the usage of prefix caching, for example when
using vllm as the inference server.

## Safety

Using tokenflood can result in high token spending. To prevent negative surprises,
we have put in place some additional safety measurements:

1. Tokenflood always tries to estimate the used tokens for the test upfront and asks you to confirm the start of the tests after seeing the estimation.
2. There are additional env variables that determine the max allowed input and output tokens for test. A test whose token usage estimate exceeds those limits will not be started.

Still, these measures do not provide perfect protection against misconfiguration. 
Always be careful when using tokenflood.

## Professional Services

Tokenflood is developed by Thomas Werkmeister, a freelance software engineer and consultant.
If you are looking for support to optimize your LLM latency, throughput, or costs reach out
at hi@tokenflood.com
