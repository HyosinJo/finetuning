import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import time
import psutil
import GPUtil
from typing import List, Dict, Tuple, Optional
import json


class HuggingFaceModelLoader:
    """HuggingFace 모델을 로드하고 관리하는 클래스"""
    
    def __init__(self):
        self.api = HfApi()
        self.loaded_models = {}  # 로드된 모델들을 저장하는 딕셔너리
        
    def load_model(self, model_id: str, device: str = "auto", 
                   torch_dtype: torch.dtype = torch.float16,
                   trust_remote_code: bool = True) -> tuple:
        """HuggingFace에서 모델과 토크나이저를 로드
        
        Args:
            model_id: HuggingFace 모델 ID
            device: 디바이스 설정 ('auto', 'cuda', 'cpu')
            torch_dtype: 모델 데이터 타입 (메모리 절약을 위해 float16 권장)
            trust_remote_code: 원격 코드 실행 허용 여부
        
        Returns:
            tuple: (model, tokenizer) 튜플
        """
        
        # 이미 로드된 모델인지 확인
        if model_id in self.loaded_models:
            print(f"모델 {model_id}가 이미 로드되어 있습니다")
            return self.loaded_models[model_id]
        
        try:
            print(f"모델 로딩 중: {model_id}")
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=trust_remote_code
            )
            
            # pad_token이 없는 경우 eos_token으로 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 모델 로드 (자동으로 적절한 디바이스에 배치)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=trust_remote_code
            )
            
            # 로드된 모델 저장
            self.loaded_models[model_id] = (model, tokenizer)
            print(f"모델 {model_id} 로드 완료!")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def generate_text(self, model_id: str, prompt: str, 
                     max_new_tokens: int = 100, 
                     temperature: float = 0.7,
                     do_sample: bool = True,
                     top_p: float = 0.9) -> str:
        """로드된 모델로 텍스트 생성
        
        Args:
            model_id: 사용할 모델 ID
            prompt: 입력 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 생성 다양성 조절 (0.0~1.0)
            do_sample: 샘플링 사용 여부
            top_p: nucleus sampling 파라미터
        
        Returns:
            str: 생성된 텍스트 (프롬프트 제외)
        """
        
        if model_id not in self.loaded_models:
            raise ValueError(f"모델 {model_id}이 로드되지 않았습니다")
        
        model, tokenizer = self.loaded_models[model_id]
        
        # 프롬프트를 토큰으로 변환
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # GPU가 사용 가능한 경우 입력을 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 텍스트 생성 (gradient 계산 비활성화로 메모리 절약)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 생성된 토큰을 텍스트로 디코딩하고 프롬프트 부분 제거
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def unload_model(self, model_id: str):
        """메모리에서 모델 제거"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            torch.cuda.empty_cache()  # GPU 메모리 정리
            print(f"모델 {model_id} 언로드 완료")


class ModelBenchmark:
    """여러 모델의 성능을 측정하고 비교하는 벤치마크 클래스"""
    
    def __init__(self):
        self.loader = HuggingFaceModelLoader()
        self.results = {}
        
    def get_gpu_memory(self):
        """현재 GPU 메모리 사용량 반환 (MB)"""
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        return 0
    
    def get_cpu_memory(self):
        """현재 CPU 메모리 사용량 반환 (GB)"""
        return psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
    
    def benchmark_model(self, model_id: str, prompts: List[str], 
                       max_new_tokens: int = 50) -> Dict:
        """단일 모델에 대한 벤치마크 실행
        
        측정 항목:
        - 모델 로딩 시간
        - 각 프롬프트별 생성 시간
        - 초당 토큰 생성 속도
        - GPU/CPU 메모리 사용량
        """
        print(f"\n{'='*50}")
        print(f"벤치마킹 중: {model_id}")
        print(f"{'='*50}")
        
        # 결과를 저장할 딕셔너리 초기화
        results = {
            'model_id': model_id,
            'load_time': 0,  # 모델 로딩 시간
            'generation_times': [],  # 각 프롬프트별 생성 시간
            'tokens_per_second': [],  # 초당 토큰 수
            'memory_usage': {},  # 메모리 사용량
            'outputs': []  # 생성된 텍스트들
        }
        
        # 초기 메모리 사용량 측정
        initial_gpu_mem = self.get_gpu_memory()
        initial_cpu_mem = self.get_cpu_memory()
        
        start_time = time.time()
        try:
            model, tokenizer = self.loader.load_model(model_id)
            load_time = time.time() - start_time
            results['load_time'] = load_time
            
            loaded_gpu_mem = self.get_gpu_memory()
            loaded_cpu_mem = self.get_cpu_memory()
            
            results['memory_usage'] = {
                'gpu_mb': loaded_gpu_mem - initial_gpu_mem,
                'cpu_gb': loaded_cpu_mem - initial_cpu_mem
            }
            
            print(f"모델 로딩 시간: {load_time:.2f}초")
            print(f"GPU 메모리 사용량: {results['memory_usage']['gpu_mb']:.2f} MB")
            print(f"CPU 메모리 사용량: {results['memory_usage']['cpu_gb']:.2f} GB")
            
            for i, prompt in enumerate(prompts):
                print(f"\n프롬프트 {i+1}: {prompt[:50]}...")
                
                gen_start = time.time()
                output = self.loader.generate_text(
                    model_id, 
                    prompt, 
                    max_new_tokens=max_new_tokens
                )
                gen_time = time.time() - gen_start
                
                num_tokens = len(tokenizer.encode(output))
                tokens_per_sec = num_tokens / gen_time
                
                results['generation_times'].append(gen_time)
                results['tokens_per_second'].append(tokens_per_sec)
                results['outputs'].append({
                    'prompt': prompt,
                    'output': output,
                    'time': gen_time,
                    'tokens_per_second': tokens_per_sec
                })
                
                print(f"생성 시간: {gen_time:.2f}초")
                print(f"초당 토큰 수: {tokens_per_sec:.2f}")
                print(f"생성된 텍스트: {output[:100]}...")
                
            avg_gen_time = sum(results['generation_times']) / len(results['generation_times'])
            avg_tokens_per_sec = sum(results['tokens_per_second']) / len(results['tokens_per_second'])
            
            results['avg_generation_time'] = avg_gen_time
            results['avg_tokens_per_second'] = avg_tokens_per_sec
            
            print(f"\n평균 생성 시간: {avg_gen_time:.2f}초")
            print(f"평균 초당 토큰 수: {avg_tokens_per_sec:.2f}")
            
        except Exception as e:
            print(f"벤치마크 중 오류 발생 ({model_id}): {e}")
            results['error'] = str(e)
            
        finally:
            self.loader.unload_model(model_id)
            
        self.results[model_id] = results
        return results
    
    def run_benchmark(self, model_ids: List[str], prompts: List[str] = None):
        if prompts is None:
            prompts = [
                "Explain quantum computing in simple terms:",
                "Write a Python function to calculate fibonacci numbers:",
                "What are the main differences between machine learning and deep learning?",
                "Translate the following to French: 'Hello, how are you today?'",
                "Generate a creative story about a robot learning to paint:"
            ]
        
        print(f"{len(model_ids)}개 모델에 대한 벤치마크 시작")
        print(f"{len(prompts)}개 프롬프트 사용")
        
        for model_id in model_ids:
            try:
                self.benchmark_model(model_id, prompts)
            except Exception as e:
                print(f"벤치마크 실패 ({model_id}): {e}")
                
        return self.results
    
    def save_results(self, filename: str = "benchmark_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n결과가 {filename}에 저장되었습니다")
    
    def print_summary(self):
        print(f"\n{'='*70}")
        print("벤치마크 요약")
        print(f"{'='*70}")
        print(f"{'모델':<30} {'로드 시간':<10} {'평균 생성':<12} {'토큰/초':<10} {'GPU MB':<10}")
        print(f"{'-'*70}")
        
        for model_id, results in self.results.items():
            if 'error' not in results:
                print(f"{model_id[:29]:<30} "
                      f"{results['load_time']:<10.2f} "
                      f"{results['avg_generation_time']:<12.2f} "
                      f"{results['avg_tokens_per_second']:<10.2f} "
                      f"{results['memory_usage']['gpu_mb']:<10.2f}")
            else:
                print(f"{model_id[:29]:<30} ERROR: {results['error'][:35]}")


if __name__ == "__main__":
    # 벤치마크 실행
    benchmark = ModelBenchmark()
    
    # 테스트할 모델 목록
    models_to_test = [
        "gpt-oss/gpt-oss-20b",  # 20B 파라미터 모델
        "DeepSeek-R1-Distill-Qwen-1.5B"
    ]
    
    # 벤치마크용 프롬프트
    custom_prompts = [
        "Explain the theory of relativity:",  # 과학 설명
        "Write a sorting algorithm in Python:",  # 코드 생성
        "What is the meaning of life?",  # 철학적 질문
        "Describe the process of photosynthesis:",  # 생물학 설명
        "Generate code to create a REST API:"  # 복잡한 코드 생성
    ]
    
    # 벤치마크 실행
    results = benchmark.run_benchmark(models_to_test, custom_prompts)
    
    # 결과 출력 및 저장
    benchmark.print_summary()
    benchmark.save_results("model_benchmark_results.json")