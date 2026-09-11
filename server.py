import base64
import logging 
import sys

from mcp.server.fastmcp import FastMCP, Image
from model_manager import ModelManager
from tools.attention import run_attention
from tools.generate import run_generate

from tools.schemas import (
    AttentionInput , 
    AttentionOutput , 
    GeneratedOutput , 
    GeneratedInput ,
)

logger = logging.getLogger("mcp_server")

mcp = FastMCP("local-model_inference")

manager = ModelManager(
    draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device = "cpu"
)

@mcp.tool()
def generate(
    prompt: str , 
    max_new_tokens : int = 100 , 
    use_speculative : bool = True , 
    k : int = 4 , 
    temp : float = 0.7,):

    pass



@mcp.tool()
def get_attention_pattern(prompt : str , layer : int , head: str = 'all'):
    try:
        args = AttentionInput(prompt=prompt, layer=layer, head=head)
    
    except Exception as e:
        return [ToolError(error="invalid_input", detail=str(e)).model_dump_json()]
    
    try:
        output = run_attention(manager = ModelManager , args = args)
    except ValueError as e:
        # Expected, structured failure mode: bad layer/head index.
        return [ToolError(error="invalid_layer_or_head", detail=str(e)).model_dump_json()]
    except Exception as e:
        logger.exception("get_attention_pattern failed")
        return [ToolError(error="attention_extraction_failed", detail=str(e)).model_dump_json()]

 



