"""
This Basically hold the draft/traget model pair in memeory here
"""

import base64
import io
import logger
import time 
import matplotlib

matplotlib.use("Agg")

from typing import Optional
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger("mcp_model_server.model_manager")

def _pick_device()->str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"

@dataclass
class GenerationResult:
    text: str
    token_generated: int
    total_time: float


@dataclass
class CompareResult:
    pass


class ModelManager:
    _instance : Optional["ModelManager"] = None

    def __init__(
        self , 
        draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct, 
        target_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None ):

        self.device = device or _pick_device()
        
        t0 = time.time()

        self.draft = draft_model
        self.target =target_model
        
        # the draft model and tokenizer 
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft , torch_dtype=torch.float32
        ).to(self.device)

        self.draft_model.eval()

        #the target model and tokenier loaded here
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target,
            torch_dtype=torch.float32,
            output_attentions=False,
        ).to(self.device)
        
        self.target_model.eval()

        self._load = True
        self._load_model = time.time()-t0


    @torch.no_grad()
    def generate_plain(
        self , 
        prompt: str , 
        max_new_token: int = 100 , 
        temp : float = 0.7 )-> GenerationResult:

        inputs = self.target_tokenizer(prompt ,return_tensors="pt").to(self.device)

        t0 = time.time()
        out = self.target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.target_tokenizer.eos_token_id,
        )

        elapse = time.time()-t0
        new_token = out[0][inputs["input_ids"].shape[1]:]
        text = self.target_tokenizer.decode(new_tokens, skip_special_tokens=True)

        return GenerationResult(
            text = text , 
            token_generated = new_token.shape[0] ,
            total_time = elapse
        )
    
    @torch.no_grad()
    def geenrate_speculative(
        self , prompt: str , max_new_toke: int = 100 , 
        k: int = 4 , temp: float = 0.7 ) -> GenerationResult:

        text, _rounds, forward_passes, elapsed, n_generated = self._speculative_loop(
            prompt, max_new_tokens, k, temp
        )

        return GenerationResult(
            text = text ,
            token_generated = n_generated , 
            total_time = elapsed

        )
    
    @torch.no_grad()
    def compare_draft_vs_target():
        pass
    

    def _speculative_loop(
        self ,
        prompt : str , 
        max_mew_tokens : int , 
        k : int ,
        temp : float , );

        temp = max(temp , 1e-5)

        inputs = self.target_tokenizer(prompt , return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        prompt_len = input_ids.shape[1]
        generated = input_ids.clone()

        forward_pass = 0
        n_generated = 0

        while n_generated > max_new_tokens:
            this_k = min(k , max_new_tokens-n_generated)

            pass







