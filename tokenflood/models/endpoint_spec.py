from typing import Dict, Optional

from pydantic import BaseModel


class EndpointSpec(BaseModel):
    model: str
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    deployment: Optional[str] = None
    extra_headers: Dict = {}
