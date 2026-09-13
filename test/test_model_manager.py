from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_manager import ModelManager

def main() -> None:
    print("Loading models (this should happen once)...")
    manager = ModelManager()
    #print(f"Loaded on device={manager.device} in {manager._load_time:.2f}s\n")
    
    prompt = "The three laws of thermodynamics are"
    print("=== generate_plain ===")
    plain = manager.generate_plain
    #print(f"text: {plain.text!r}")
    #print(f"total tokens/sec: {plain.total_time:.2f}\n")
    
    print("=== generate_speculative ===")
    spec = manager.geenrate_speculative(prompt= prompt)

    #print(f"text : {spec.text}")

    print("=== compare_draft_vs_target ===")
    cmp = manager.compare_draft_vs_target(prompt = prompt)
    
    print("=== get_attention_pattern ===")
    png_64 , matrix , tokens = manager.get_attention_patter(prompt, layer=0, head="all")
    #print(f"tokens: {tokens}")
    #print(f"attention matrix shape: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
    #print(f"png base64 length: {len(png_b64)} chars")

    print("\nAll checks ran without raising. Inspect the printed output above by eye.")

if __name__ == "__main__":
    main()





