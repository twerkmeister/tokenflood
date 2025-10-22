from tokenflood.constants import REQUESTS_PER_SECOND_COLUMN_NAME


def test_as_dataframe(tiny_run_data):
    df = tiny_run_data[0].as_dataframe()
    assert len(df) == 2
    assert REQUESTS_PER_SECOND_COLUMN_NAME in df.columns
    for field in tiny_run_data[0].results.model_dump().keys():
        assert field in df.columns
