import pytest


def test_endpoint_spec_model_id_folder_name(base_endpoint_spec):
    assert (
        base_endpoint_spec.folder_name
        == "hosted_vllm_HuggingFaceTB_SmolLM_135M_Instruct"
    )


@pytest.mark.parametrize(
    "name, expected_result",
    [
        ("abc", "abc"),
        ("123", "123"),
        ("abc-123_", "abc_123_"),
        ("ab/cd", "ab_cd"),
        ("ax.exe", "ax_exe"),
        ("!\"'#$%^*()[]?", "_____________"),
    ],
)
def test_endpoint_spec_folder_name_sanitization(
    base_endpoint_spec, name, expected_result
):
    endpoint_spec = base_endpoint_spec.model_copy(update={"name": name})
    assert endpoint_spec.folder_name == expected_result
