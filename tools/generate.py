from model_manager import ModelManager
from tools .schemas import GeneratedInput , GeneratedOutput


def run_generate(manager: ModelManager , args: GeneratedInput)->GeneratedOutput:
    if args.use_speculative:
        result = manager.geenrate_speculative(
            prompt = args.prompt , 
            max_new_token = args.max_new_tokens , 
            k = args.k , 
            temp = args.temperature
        )
    
    else:
        result = manager.generate_plain(
             prompt: str = args.prompt , 
            max_new_token: int = args.max_new_tokens , 
             temp : float = args.temperature)
    

    return GeneratedOutput(
        text = result.text
    )
    