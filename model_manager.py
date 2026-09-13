from __future__ import annotations

"""
This Basically hold the draft/traget model pair in memeory here
"""

import base64
import io
import logging
import time 
import matplotlib
from dataclasses import dataclass, field

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
    output: torch.Tensor


@dataclass
class CompareResult:
    generated_text: str
    total_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    run_length_distribution: list[int]
    rounds: list[SpecRoundLog]
    target_forward_passes: int


@dataclass
class SpecRoundLog:
    round_index : int
    proposed_token : list[str]
    accepted_mask : list[bool]
    p_values : list[float]
    q_values : list[float]
    run_length : int


class ModelManager:
    _instance : Optional["ModelManager"] = None

    def __init__(
        self , 
        draft_model: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
        target_model: str = 'HuggingFaceTB/SmolLM2-360M-Instruct',
        device: Optional[str] = None , ):

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

        print("both the model has been downloaded")

    """Generate a plain answer here """
    @torch.no_grad()
    def generate_plain(
        self , 
        prompt: str , 
        max_new_token: int = 100 , 
        temp : float = 0.7 )-> GenerationResult:

        if getattr(self.target_tokenizer, "chat_template", None):
            formatted_prompt = self.target_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        inputs = self.target_tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        t0 = time.time()
        out = self.target_model.generate(
            **inputs,
            max_new_tokens=max_new_token,
            do_sample=temp > 0,
            temperature=temp if temp > 0 else 1.0,
            pad_token_id=self.target_tokenizer.eos_token_id,
        )

        elapse = time.time()-t0
        new_token = out[0][inputs["input_ids"].shape[1]:]
        text = self.target_tokenizer.decode(new_token, skip_special_tokens=True)
        output_text = self.target_tokenizer.decode(out , skip_special_tokens = True)

        return GenerationResult(
            text = text , 
            token_generated = new_token.shape[0] ,
            total_time = elapse , 
            output = output_text
        )
    ##################################################################################
    @torch.no_grad()
    def geenrate_speculative(
        self , prompt: str , max_new_tokens: int = 100 , 
        k: int = 4 , temp: float = 0.7 ):

        text, _rounds, forward_passes, elapsed, n_generated = self._speculative_loop(
            prompt, max_new_tokens, k, temp
        )

        """NEED TO MIDIFY HERE"""
        return [text, _rounds, forward_passes, elapsed, n_generated]
    
    ###################################################################################

    @torch.no_grad()
    def compare_draft_vs_target(
        self,
        prompt: str,
        k: int = 4,
        max_new_token: int = 40,
        temp: float = 0.7,

    ) -> CompareResult:
        text, rounds, forward_passes, elapsed, n_generated = self._speculative_loop(
            prompt, max_new_token, k, temp
        )
        run_lengths = [r.run_length for r in rounds]
        accepted = sum(run_lengths)
        rejected = n_generated - accepted

        print("##############the rounds here is " , rounds)
        
        return CompareResult(
            generated_text=text,
            total_tokens=n_generated,
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            acceptance_rate=accepted / n_generated if n_generated else 0.0,
            run_length_distribution=run_lengths,
            rounds=rounds,
            target_forward_passes=forward_passes,
        )


    
    """ get the attention pattern here"""
    @torch.no_grad()
    def get_attention_patter(self , prompt: str , layer: int , head: str | int = "all" ):
        inputs = self.target_tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        # SDPA cannot return attention weights; eager attention is required here.
        self.target_model.set_attn_implementation("eager")
        out = self.target_model(**inputs, output_attentions=True)

        attentions = out.attentions #tuple (batch , n_head , seq , seq ) here

        if not attentions:
            print(" the function doesnot return any atttention output here") 

        n_layers = len(attentions)

        if not(0<=layer<n_layers):
            raise ValueError(f"The layer number is not valid here and the number of the layer is {n_layers}")
        
        layer_attn = attentions[layer][0] # the dimension here is [n_heads, seq, seq]
        n_heads = layer_attn.shape[0]

        if head == "all":
            matrix = layer_attn.mean(dim=0)
            head_label = "avg_of_all_heads"
        else:
            head_int = int(head)
            if not (0<=head_int< n_heads):
                raise ValueError(f"the head number is not valid here and the number of heads are {n_heads}")
            
            matrix = layer_attn[head_int]
            head_label = f"head_no - {head_int}"
        
        tokens = self.target_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        matrix_np = matrix.detach().cpu().float().numpy()


        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix_np, cmap="viridis")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_title(f"Layer {layer} / {head_label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        png = base64.b64encode(buf.read()).decode("utf-8")

        return png, matrix_np.tolist(), tokens

    def _speculative_loop(
        self ,
        prompt : str , 
        max_new_tokens : int , 
        k : int ,
        temp : float , 
        collect_logs: bool = True
        ):

        #k : How many tokens the draft model proposes
        #per speculative round

        #max_new_tokens : How many tokens the FINAL output can contain overall
        #n_generated : how many token the model will generate here

        temp = max(temp , 1e-5)

        inputs = self.target_tokenizer(prompt , return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        """ What is the length of the prompt here"""
        prompt_len = input_ids.shape[1]
        rounds : list[SpecRoundLog] = []

        generated = input_ids.clone()

        forward_pass = 0
        n_generated = 0

        t0 = time.time()

        while n_generated < max_new_tokens:
            this_k = min(k, max_new_tokens - n_generated)
            cur = generated

            proposed_token_probab = []
            proposed_token = []

            """ Generate the no of token for a draft model that is no od this_k here"""
            for _ in range(this_k):
                draft_out = self.draft_model(cur)
                logits = draft_out.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                """ sampling here based on the probability here"""
                token = torch.multinomial(probs , num_samples =1)

                proposed_token_probab .append(probs[0])

                proposed_token.append(token)
                cur = torch.cat([cur , token] , dim = 1)
            
            """ feeding the output into the target model here"""
            target_out = self.target_model(cur)
            forward_pass += 1


            base = generated.shape[1]-1
            target_logits = target_out.logits[0, base : base + this_k + 1, :] / temp
            target_probs = F.softmax(target_logits, dim=-1)
            

            accepted_mask = []
            p_values = []
            q_values = []
            proposed_strs = []
            run_length = 0
            bonus_token_str = None
            all_accepted = True

            for i in range(this_k):
                x_i = proposed_token[i]
                p_i = target_probs[i]

                q_i = proposed_token_probab [i]

                p_x = p_i[x_i.item()].item()
                q_x = q_i[x_i.item()].item()
                accept_prob = min(1.0, p_x / q_x) if q_x > 0 else 0.0

                if collect_logs:
                    proposed_strs.append(self.target_tokenizer.decode(x_i[0]))
                    p_values.append(p_x)
                    q_values.append(q_x)
                

                if torch.rand(1).item() < accept_prob:
                    generated = torch.cat([generated, x_i], dim=1)
                    n_generated += 1
                    run_length += 1
                    accepted_mask.append(True)
                    if n_generated >= max_new_tokens:
                        all_accepted = False  # stop, no bonus token needed
                        break

                else:
                    accepted_mask.append(False)
                    adjusted = torch.clamp(p_i - q_i, min=0.0)
                    denom = adjusted.sum()
                    if denom > 0:
                        adjusted = adjusted / denom
                        x_new = torch.multinomial(adjusted, num_samples=1).unsqueeze(0)
                    else:
                        x_new = torch.multinomial(p_i, num_samples=1).unsqueeze(0)

                    generated = torch.cat([generated, x_new], dim=1)
                    n_generated += 1
                    all_accepted = False
                    break
            

            # print("the p_value here is " , p_values)
            # print("the q_value here is " , q_values)
            # print("the acceptance mask here is " , accepted_mask)
            # print("the proposed token here is " , proposed_strs)


            if all_accepted and n_generated < max_new_tokens:
                bonus_prob = target_probs[this_k]
                bonus_token = torch.multinomial(bonus_prob, num_samples=1).unsqueeze(0)
                generated = torch.cat([generated , bonus_token] , dim = 1)
                n_generated += 1
                run_length += 1

                if collect_logs:
                    bonus_token_str = self.target_tokenizer.decode(bonus_token[0])
            

            if collect_logs:
                rounds.append(
                    SpecRoundLog(
                        round_index = 1,
                        proposed_token = proposed_strs,
                        accepted_mask = accepted_mask,
                        p_values = p_values , 
                        q_values = q_values ,
                        run_length = run_length,
                    )
                )
                
        
        elasped = time.time()-t0
        text = self.target_tokenizer.decode(
            generated[0][prompt_len:], skip_special_tokens=True
        )

        return text, rounds, forward_pass, elasped, n_generated

def main():
    manager = ModelManager()
    ans = manager.compare_draft_vs_target( prompt = "what is the capital of japan?" )

    print("the answer here is " , ans)

if __name__ == "__main__":
    main()