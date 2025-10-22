from typing import Dict, Optional

from pydantic import BaseModel


class EndpointSpec(BaseModel):
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    deployment: Optional[str] = None
    extra_headers: Dict = {}

    @property
    def provider_model_str_as_folder_name(self) -> str:
        return self.provider_model_str.replace("/", "__")

    @property
    def provider_model_str(self) -> str:
        return f"{self.provider}/{self.model}"
