from model_manager import ModelManager
from tools.schemas import CompareInput , CompareOutput

def run_compare(manager: ModelManager , args:CompareInput)->CompareOutput:

    result = ModelManager.compare_draft_vs_target(
        prompt = manager.prompt ,
        k = manager.k ,
        max_new_token = manager.max_new_tokens , 
        temp = manager.temperature
    )

    return CompareOutput(
        generated_text = result.generated_text ,
        total_tokens = result.n_generated ,
        accepted_tokens = result.accepted , 
        rejected_tokens = result.rejected ,
        acceptance_rate = result.acceptance_rate , 
        run_length_distribution = result.run_lengths ,
        rounds= result.rounds , 
        target_forward_passes = result.target_forward_passes
    )

