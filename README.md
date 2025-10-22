# Tokenflood

Tokenflood is a load testing tool for instruction-tuned LLMs that allows you to 
run arbitrary load profiles without needing specific prompt and response data.
**Define desired prompt lengths, prefix lengths, output lengths and request rates, 
and tokenflood simulates this workload for you.** 

Tokenflood uses [litellm](https://www.litellm.ai/) under the hood and supports 
[all providers that litellm covers](https://docs.litellm.ai/docs/providers).

> [!CAUTION]
> Tokenflood can generate high costs if configured poorly and used with pay-per- 
> token services. Make sure you only test workloads that are within a reasonable budget.

## Installation

```bash
pip install tokenflood
```

## Quick Start

```bash
# This creates config files for a tiny first run: run.yml and endpoint.yml
tokenflood ripples
# Afterwards you can inspect those files and run them
tokenflood run run.yml endpoint.yml

```

## Endpoint Examples

* Self-hosted VLLM
* Openai
* Bedrock
* Azure
* Gemini
