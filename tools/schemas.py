from typing import Optional 
from pydantic import BaseModel, Field

#------------------------------------------
#Generated value here
#------------------------------------------

class GeneratedInput(BaseModel):
    prompt : str
    max_new_tokens : int  = Field(default=100, ge=1, le=1024)
    use_speculative : bool = True 
    k: int = Field(default=4, ge=1, le=16, description="Draft length per round")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class GeneratedOutput(BaseModel):
    text : str
    tokens_generated : int 
    tokens_per_second : float
    target_forward_passes: int
    used_speculative: bool

"""
Gen attention pattern here
"""
class AttentionInput(BaseModel):
    prompt : str
    layer : int = Field(ge = 0)
    head: str = Field(default="all", description="Head index as a string, or 'all' to average")

class AttentionOutput(BaseModel):
    png_img: str
    attention_weights: list[list[float]]
    tokens: list[str]
    layer: int
    head: str
    given_prompt: str
  
"""
Structure error here
"""

class ToolError(BaseModel):
    error: str
    detail : str

"""
Compare draft vs target model here
"""

class CompareInput(BaseModel):
    prompt: str
    k: int = Field(default=4, ge=1, le=16)
    max_new_tokens: int = Field(default=40, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class CompareOutput(BaseModel):
    pass