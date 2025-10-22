from datetime import datetime

from tokenflood.models.endpoint_spec import EndpointSpec


def get_run_name(endpoint_spec: EndpointSpec):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{date_str}_{endpoint_spec.provider_model_str_as_folder_name()}"
