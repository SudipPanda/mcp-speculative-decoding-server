from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = PROJECT_ROOT / "server.py"

EXPECTED_TOOLS  = {'generate' , 'get_attention_pattern' , 'comparer_decoder'}

async def check_tools(session:ClientSession):
    result = await session.list_tool()
    found = {t.name for t in result.tools}
    missing = EXPECTED_TOOLS - found

    if missing: 
        print("wrong tools are being fetch ")
        sys.exit(1)

async def check_generate(session:ClientSession):
    result = await session.call_tool(
        "generate",
        {
            "prompt": "The three laws of thermodynamics are",
            "max_new_tokens": 8,
            "use_speculative": True,
            "k": 4,
            "temperature": 0.7,
        },
    )


