from model_manager import ModelManager
from tools.schemas import AttentionInput , AttentionOutput


def run_attention(manager: ModelManager , args: AttentionInput) -> AttentionOutput:
    png_img  , matrics , tokens = ModelManager.get_attention_patter(
        prompt = args.prompt , layer =  args.layer , head = 
        args.head
    )

    return AttentionOutput(
        png_img = png_img , 
        attention_weights = matrics , 
        tokens = tokens ,
        layer = args.layer , 
        head = args.head , 
        given_prompt = args.prompt
    )