from typing import Dict, Optional, Literal
import re

from pydantic import BaseModel


class EndpointSpec(BaseModel):
    provider: str
    model: str
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    deployment: Optional[str] = None
    extra_headers: Dict = {}
    extra_body: Dict = {}
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None
    ) = None

    @property
    def folder_name(self) -> str:
        name = self.name if self.name else self.provider_model_str
        return re.sub("\W", "_", name)

    @property
    def provider_model_str(self) -> str:
        return f"{self.provider}/{self.model}"
