#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

model_path = "/Users/ai/llm_proj/finetune_MobileLLM-600M_GRPO_LoRA_KMMLU/checkpoint-180"

def convert_to_gguf(model_path, output_dir=None, quantization="f16"):
    """
    Convert HuggingFace model to GGUF format for llama.cpp
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)
    
    # Check if it's a LoRA adapter
    is_lora = (model_path / "adapter_config.json").exists()
    
    if is_lora:
        print("✓ Detected LoRA adapter checkpoint")
        # Need to merge with base model first
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        # Read adapter config to get base model
        import json
        with open(model_path / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path", "facebook/MobileLLM-600M")
        
        print(f"  Base model: {base_model_id}")
        print("  Loading and merging...")
        
        # Create temporary merged directory
        merged_dir = model_path.parent / "temp_merged_model"
        merged_dir.mkdir(exist_ok=True)
        
        # Load and merge
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        model.save_pretrained(merged_dir)
        
        # Modify config to pretend it's Llama for GGUF conversion
        import json
        config_path = merged_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Change model_type to llama
        config["model_type"] = "llama"
        config["architectures"] = ["LlamaForCausalLM"]
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save tokenizer - handle MobileLLM special case
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            if isinstance(tokenizer, bool):
                # MobileLLM returns bool, use LlamaTokenizer directly
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(base_model_id)
            tokenizer.save_pretrained(merged_dir)
        except:
            # Copy tokenizer files from adapter checkpoint
            import shutil
            for tokenizer_file in ["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
                src = model_path / tokenizer_file
                if src.exists():
                    shutil.copy(src, merged_dir / tokenizer_file)
        
        # Update model_path to merged directory
        model_path = merged_dir
        print("✓ LoRA merged with base model")
    
    # Set output directory
    if output_dir is None:
        output_dir = model_path.parent / "gguf_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Clone llama.cpp if not exists
    llama_cpp_dir = Path("./llama.cpp")
    if not llama_cpp_dir.exists():
        print("Cloning llama.cpp repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)
    
    # Build llama.cpp if not built
    build_dir = llama_cpp_dir / "build"
    quantize_exe = build_dir / "bin" / "llama-quantize"
    main_exe = build_dir / "bin" / "llama-cli"
    
    if not quantize_exe.exists():
        print("Building llama.cpp with CMake...")
        build_dir.mkdir(exist_ok=True)
        
        # Configure with CMake
        subprocess.run([
            "cmake", "..", 
            "-DCMAKE_BUILD_TYPE=Release"
        ], cwd=str(build_dir), check=True)
        
        # Build
        subprocess.run([
            "cmake", "--build", ".", "--config", "Release"
        ], cwd=str(build_dir), check=True)
    
    # Install required dependencies
    print("Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "transformers", "sentencepiece", "protobuf",
        "mistral-common", "tiktoken", "gguf"
    ], check=True)
    
    # Convert to GGUF format
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    output_file = output_dir / f"model-{quantization}.gguf"
    
    print(f"Converting model to GGUF format...")
    convert_cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(output_file),
        "--outtype", quantization
    ]
    
    try:
        subprocess.run(convert_cmd, check=True)
        print(f"✓ Model converted successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
    
    # Additional quantization options
    if quantization != "f16" and quantization != "f32":
        print(f"Quantizing to {quantization}...")
        quantized_file = output_dir / f"model-{quantization}.gguf"
        subprocess.run([
            str(quantize_exe),
            str(output_file),
            str(quantized_file),
            quantization.upper()  # llama-quantize expects uppercase
        ], check=True)
    
    # Create example run script
    run_script = output_dir / "run_model.sh"
    with open(run_script, "w") as f:
        f.write(f"""#!/bin/bash
# Run the converted model
{main_exe} -m {output_file} \\
    -n 256 \\
    -p "한국의 수도는" \\
    --temp 0.7 \\
    --top-p 0.9 \\
    --repeat-penalty 1.1
""")
    os.chmod(run_script, 0o755)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Model file: {output_file}")
    print(f"  Run script: {run_script}")
    
    # Clean up temporary merged directory if it was created
    if is_lora and (model_path.parent / "temp_merged_model").exists():
        import shutil
        shutil.rmtree(model_path.parent / "temp_merged_model")
        print("✓ Cleaned up temporary files")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to GGUF for llama.cpp")
    parser.add_argument("model_path", help="Path to the model directory")
    parser.add_argument("-o", "--output", help="Output directory (default: model_path/gguf_output)")
    parser.add_argument("-q", "--quantization", default="f16", 
                       choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                       help="Quantization type (default: f16)")
    
    args = parser.parse_args()
    
    # Convert model
    convert_to_gguf(args.model_path, args.output, args.quantization)

if __name__ == "__main__":
    # If no arguments provided, use default path
    if len(sys.argv) == 1:
        model_path = model_path
        print(f"Using default model path: {model_path}")
        convert_to_gguf(model_path)
    else:
        main()