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
> See the safety section for more information.

## Common Usage Scenarios

1. Load testing self-hosted LLMs.
2. Assessing the effects of hardware, quantization, and prompt optimizations on latency, throughput, and costs.
3. Assessing the intraday latency variations of hosted LLM providers for your load types.
4. Assessing and choosing a hosted LLM provider before going into production with them. 

### Example: Assessing the effects of prompt optimizations 
Here you can see an example of optimizing the prompt parameters for latency and throughput:

![An example of trying to optimize the prompt style for latency and throughput](/images/self-hosted_combined.png)

We start with the base case scenario (top left panel) and introduce two improvements:
* Increasing the number of prefix tokens, e.g. by reordering parts of the prompt. (top right panel)
* Reducing the number of output tokens, e.g. by changing the output format from verbose JSON to a custom DSL (bottom left pannel)

Finally, the bottom right panel shows both improvements active at the same time. All tests were run on the same hardware and model.

Here's a brief extract of the data in tabular form:

| Scenario           | #Input Tokens | #Prefix Tokens | #Output Tokens | #Mean Latency in ms @ 3 requests per second | 
|--------------------|---------------|----------------|----------------|---------------------------------------------|
| base case          | 3038          | 1000           | 60             | 1758                                        |
| more prefix tokens | 3038          | 2000           | 60             | 1104                                        |
| shorter output     | 3038          | 1000           | 30             | 921                                         |
| both changes       | 3038          | 2000           | 30             | 602                                         |


## Professional Services

If you are looking for professional support to
* optimize your LLM accuracy, latency, throughput, or costs
* fine tune open models for your use case, 

feel free to reach out to me at thomas@werkmeister.me.

## Installation

```bash
pip install tokenflood
```

## Quick Start

For a quick start, make sure that vllm is installed, and you serve a small model:
```bash
pip install vllm
vllm serve HuggingFaceTB/SmolLM-135M-Instruct
```

Afterward, create the basic config files and do a first run:
```bash
# This creates config files for a tiny first run: run.yml and endpoint.yml
tokenflood init
# Afterwards you can inspect those files and run them
tokenflood run run_suite.yml endpoint.yml
```

Finally, in the `results` folder you should find your run folder
containing:
* a graph visualizing the latency quantiles across the difference request rates and the network latency (`latency_quantiles.png`)
* the raw data points collected from the LLM calls (`llm_requests.csv`)
* the raw data points collected from assessing network latency (`network_latency.csv`)
* a summary file containing lots of information about the run (`summary.yml`)
* the original run suite config used for the run (`run_suite.yml`)
* the original endpoint config used for the run (`endpoint_spec.yml`)
* an error log (`error.log`)

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
To get a better understanding, it's best to have a look at [the official documentation of the litellm completion call](https://docs.litellm.ai/docs/completion/input). 

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
model: gpt-4o-mini
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
seconds and the type of loads that are being sent.

Here is the run suite that is being created for you upon calling `tokenflood init`:
```yaml
name: ripple
requests_per_second_rates:  # Defines the phases with the different request rates
- 1.0
- 2.0
test_length_in_seconds: 10  # each phase is 10 seconds long
load_types:                 # This run suite has two load types with equal weight
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
input_token_budget: 100000  # the maximum number of input tokens this test is allowed to use - prevents any load configuration that would use more than this from starting
output_token_budget: 10000  # the maximum number of output tokens this test is allowed to use - prevents any load configuration that would use more than this from starting
```

## Heuristic Load Testing

Tokenflood does not need specific prompt data to run tests. Instead, it only needs
metadata about the prompt and task: prompt length, prefix length and output lengths. 
All counted in tokens. This allows for swift testing of alternative configurations and 
loads. Changing the token counts in the load types is a matter of seconds as opposed 
to having to adjust implementations and reobserving prompts of a system. Additionally, 
you can make sure to get exactly the desired output profile across all models and 
configurations, allowing for direct comparison between them.

### How it works

Tokenflood uses sets of strings that correspond to a single token in most tokenizers,
such as a space plus a capital letter. Sampling from this set of single token strings, 
tokenflood generates the input prompt. The defined prefix length will be non-random.
Finally, a task that usually generates a long answer is appended. In combination with
setting the maximum completion tokens for generation, tokenflood achieves the desired
output length.

### Why it works

This type of heuristic testing creates reliable data because the processing time of an
non-reasoning LLM only depends on the length of input and output and any involved caching 
mechanisms.

### Failures of the heuristic

Heuristic load testing comes with the risk of not perfectly achieving the desired token 
counts for specific models. If that happens, tokenflood will warn you during a run if 
any request diverges more than 10% from the expected input or output token lengths. At 
the end of a run, you will also be warned about the average divergence if it is more 
than 10% from the expected token count. In the summary file of a run, you can see the 
absolute and relative divergences again.

> [!IMPORTANT]
> You can specify the prefix length, however, whether the prefix is used will depend on the 
> specific endpoint and its configuration. Some providers, like openai, will only start to
> use prefix caching once your total prompt length exceeds 1024 tokens. Additionally,
> it seems litellm does not always record the usage of prefix caching. When
> using vllm as the inference server, it never reports any cached tokens. At the same 
> time, one can see a big difference in latency between using and not using prefix 
> caching despite the cached tokens not being reported properly. Due to this issue,
> tokenflood currently does not warn when the desired prefix tokens diverge from the 
> measured ones.


## Safety

Using tokenflood can result in high token spending. To prevent negative surprises,
tokenflood has additional safety measurements:

1. Tokenflood always tries to estimate the used tokens for the test upfront and asks you to confirm the start of the tests after seeing the estimation.
2. There are additional run suite variables that determine the maximum allowed input and output token budget for the test. A test whose token usage estimate exceeds those limits will not be started.

Still, these measures do not provide perfect protection against misconfiguration. 
Always be careful when using tokenflood.
