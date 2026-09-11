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
             prompt = args.prompt ,
            max_new_token = args.max_new_tokens , 
             temp = args.temperature)
    

    return GeneratedOutput(
        text=result.text,
        tokens_generated=result.token_generated,
        tokens_per_second=(
            result.token_generated / result.total_time
            if result.total_time > 0 else 0.0
        ),
        target_forward_passes=0,
        used_speculative=args.use_speculative,
    )
    