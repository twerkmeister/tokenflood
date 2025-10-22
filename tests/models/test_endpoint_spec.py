def test_endpoint_spec_model_id_as_folder_name(base_endpoint_spec):
    assert (
        base_endpoint_spec.provider_model_str_as_folder_name()
        == "hosted_vllm__HuggingFaceTB__SmolLM-135M-Instruct"
    )
