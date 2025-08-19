import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer 병렬 처리 경고 방지
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 불필요한 경고 메시지 숨김

import torch
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
from torch.fx import symbolic_trace
from torch._dynamo import optimize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import numpy as np
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import json
import psutil
import GPUtil
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
import random
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
import nltk
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# 비교할 모델들 설정
#'trillionlabs/Tri-7B'
base_model_id = 'openai/gpt-oss-20b' #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 기존 모델
fine_model_id = "/Users/ai/llm_proj/finetune_gpt-oss-20b_SFT_LoRA_heegyu/CoT-collection-ko/checkpoint-1890"  # 파인튜닝 모델

# 개별 실행 시 사용할 모델 (기본값)
model_id = base_model_id
data_count = 20  # 각 주제별 데이터 개수


# 사용자가 선택할 수 있는 최적화 기법들
OPTIMIZATION_METHODS = [
    "baseline",      # 최적화 없음
    "pruning",       # 가중치 프루닝 (PyTorch 내장)
    "graph_optimize",# 그래프 최적화 (PyTorch 내장)
    "torch_compile", # torch.compile 최적화 (PyTorch 2.0+)
    "mixed_precision", # 혼합 정밀도 (PyTorch 내장)
    "dynamic_quantize", # 동적 양자화 (PyTorch 내장)
    "quanto_int8",   # Quanto 8비트 양자화 (Mac 호환)
    "quanto_int4",   # Quanto 4비트 양자화 (Mac 호환)
]

# 사용자가 선택한 최적화 방법들 (순차적으로 적용)
SELECTED_OPTIMIZATIONS = ['pruning']


# KMMLU 전체 주제 목록 (45개)
# 인문학 (Humanities)
# - Korean_History: 한국사 - 고대부터 현대까지 한국 역사
# - World_History: 세계사 - 세계 문명과 역사적 사건
# - Philosophy: 철학 - 동서양 철학 사상
# - Moral: 도덕/윤리 - 윤리학과 도덕적 판단
# - Language_and_Media: 언어와 매체 - 언어학, 커뮤니케이션
# - Law: 법학 - 헌법, 민법, 형법 등

# 사회과학 (Social Sciences)  
# - Political_Science_and_Sociology: 정치학/사회학 - 정치 체제, 사회 구조
# - Economics: 경제학 - 미시/거시경제학
# - Psychology: 심리학 - 인지, 발달, 사회심리학
# - Geography: 지리학 - 자연/인문지리
# - Education: 교육학 - 교육 이론과 방법
# - Social_Welfare: 사회복지학 - 복지 정책과 실천

# STEM - 수학/과학 (Mathematics & Sciences)
# - Math: 수학 - 대수, 기하, 미적분, 통계
# - Physics: 물리학 - 역학, 전자기학, 현대물리 (KMMLU에는 없음 - Mechanical-Engineering, Electrical-Engineering으로 대체)
# - Chemistry: 화학 - 일반/유기/무기화학  
# - Biology: 생물학 - 세포생물학, 유전학, 생태학
# - Earth_Science: 지구과학 - 지질학, 기상학, 천문학

# STEM - 공학/기술 (Engineering & Technology)
# - Computer_Science: 컴퓨터공학 - 알고리즘, 자료구조, AI
# - Electrical_Engineering: 전기공학 - 회로, 전자기학
# - Mechanical_Engineering: 기계공학 - 역학, 열역학
# - Civil_Engineering: 토목공학 - 구조, 수리, 토질
# - Chemical_Engineering: 화학공학 - 공정, 열역학
# - Industrial_Engineering: 산업공학 - 최적화, 품질관리

# 의학/보건 (Medicine & Health)
# - Clinical_Medicine: 임상의학 - 내과, 외과 등
# - Korean_Medicine: 한의학 - 한의학 이론과 실제
# - Nursing: 간호학 - 간호 이론과 실습
# - Public_Health: 보건학 - 역학, 보건정책
# - Pharmacy: 약학 - 약리학, 약제학

# 경영/경제 (Business & Economics)
# - Business_Administration: 경영학 - 경영전략, 마케팅
# - Accounting: 회계학 - 재무/관리회계
# - Marketing: 마케팅 - 소비자행동, 브랜드
# - International_Trade: 무역학 - 국제무역 이론과 실무
# - Finance: 재무금융 - 투자, 기업재무

# 기타 전문 분야 (Other Professional Fields)
# - Agricultural_Science: 농업과학 - 작물학, 원예학
# - Architecture: 건축학 - 건축 설계와 이론
# - Fashion: 의류학 - 패션 디자인과 산업
# - Food_Science: 식품과학 - 영양학, 식품공학
# - Environmental_Science: 환경과학 - 환경 오염과 보전
# - Maritime_Science: 해양과학 - 해양학, 수산학
# - Military_Science: 군사학 - 전략, 전술
# - Aviation: 항공학 - 항공기 원리와 운항
# - Railroad: 철도공학 - 철도 시스템
# - Telecommunications: 정보통신 - 통신 이론과 네트워크

# 벤치마크용 주제 선택 (수학, 과학, 코딩, 일반지능)
data_subject = ["Math"]


@dataclass
class BenchmarkResult:
    """벤치마크 결과를 저장하는 데이터 클래스"""
    model_name: str
    optimization: str
    load_time: float
    inference_times: List[float]
    memory_usage: Dict[str, float]
    tokens_per_second: float
    model_size_mb: float
    accuracy: float = 0.0  # 정답률
    predictions: List[Dict] = None  # 각 문제별 예측 결과
    # 텍스트 생성 품질 지표
    bleu_score: float = 0.0  # BLEU 점수 (0-1)
    rouge_scores: Dict[str, float] = None  # ROUGE-1, ROUGE-2, ROUGE-L
    perplexity: float = 0.0  # Perplexity (낮을수록 좋음)
    coherence_score: float = 0.0  # 일관성 점수
    diversity_score: float = 0.0  # 다양성 점수
    

class LLMOptimizationBenchmark:
    """LLM 최적화 기법을 적용하고 벤치마크하는 클래스"""
    
    def __init__(self, model_id: str = "", benchmark_dataset: str = "kmmlu", subject: str = None):
        self.model_id = model_id
        self.results = {}
        self.benchmark_dataset = benchmark_dataset
        self.subject = subject
        
        # 벤치마크 데이터셋 로드 (주제가 지정된 경우 해당 주제만)
        if subject:
            self.load_benchmark_data(subject)
        
        # PyTorch 버전 확인
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
    
    def load_benchmark_data(self, subject=None):
        """벤치마크 데이터셋 로드 - 특정 주제만 로드 가능"""
        if subject:
            print(f"\n벤치마크 데이터셋 로드 중: {self.benchmark_dataset} - 주제: {subject}")
        else:
            print(f"\n벤치마크 데이터셋 로드 중: {self.benchmark_dataset}")
        
        if self.benchmark_dataset == "kmmlu":
            # KMMLU (Korean Massive Multitask Language Understanding) - 한국어 45개 주제
            try:
                # KMMLU는 각 주제별로 개별 로드 필요
                self.test_prompts = []
                subjects = [subject] if subject else data_subject
                
                for subj in subjects:
                    try:
                        # KMMLU 주제명 매핑
                        kmmlu_subject = subj  # 이미 KMMLU 형식
                        dataset = load_dataset("HAERAE-HUB/KMMLU", kmmlu_subject, split="test")
                        
                        # data_count만큼 테스트 데이터 순서대로 추출
                        # 테스트 데이터의 마지막 n개 사용 (가장 어려운 문제들이 보통 뒤에 있음)
                        dataset_list = list(dataset)
                        samples = dataset_list[-data_count:] if len(dataset_list) >= data_count else dataset_list
                        for sample in samples:
                            prompt = f"### 지시:\n다음 질문을 읽고 올바른 답을 A, B, C, D 중에서 하나만 선택하세요.\n\n### 질문:\n{sample['question']}\n\n### 선택지:\nA) {sample['A']}\nB) {sample['B']}\nC) {sample['C']}\nD) {sample['D']}\n\n### 응답:\n정답은"
                            # KMMLU는 answer가 1,2,3,4 형태이므로 A,B,C,D로 변환
                            answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
                            self.test_prompts.append({
                                'prompt': prompt,
                                'answer': answer_map.get(sample['answer'], str(sample['answer'])),
                                'subject': subj,
                                'choices': {
                                    'A': sample['A'],
                                    'B': sample['B'],
                                    'C': sample['C'],
                                    'D': sample['D']
                                }
                            })
                    except Exception as e:
                        print(f"  {subj} 주제 로드 실패: {e}")
                        continue
                
                if not self.test_prompts:
                    raise Exception("KMMLU에서 문제를 로드할 수 없습니다.")
                    
                print(f"KMMLU에서 {len(self.test_prompts)}개 문제 로드 완료")
                
                # 처음 3개 샘플 출력
                print("\n📋 KMMLU 데이터 샘플:")
                print("="*80)
                for i, sample in enumerate(self.test_prompts[:3]):
                    print(f"\n[샘플 {i+1}]")
                    print(f"프롬프트: {sample['prompt']}")
                    print(f"\n참조 응답: {sample['answer']}")
                print("="*80)
                
            except Exception as e:
                print(f"KMMLU 로드 실패: {e}")
                # MMLU로 폴백
                self.benchmark_dataset = "mmlu"
                dataset = load_dataset("cais/mmlu", "all", split="test")
                # 각 주제에서 샘플 추출
                self.test_prompts = []
                subjects = [subject] if subject else data_subject
                
                for subj in subjects:
                    # KMMLU 주제명을 MMLU 형식으로 변환 (Math -> college_mathematics 등)
                    mmlu_subject = {
                        'Math': 'college_mathematics',
                        'Physics': 'college_physics', 
                        'Computer_Science': 'computer_science',
                        'Philosophy': 'philosophy'
                    }.get(subj, subj.lower())
                    
                    subject_data = [d for d in dataset if d['subject'] == mmlu_subject]
                    if subject_data:
                        # data_count만큼 샘플 추출 (최대값 제한)
                        samples = random.sample(subject_data, min(data_count, len(subject_data)))
                        for sample in samples:
                            prompt = f"Question: {sample['question']}\n\nChoices:\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}\n\nPlease answer with just the letter (A, B, C, or D). Answer:"
                            # MMLU는 answer가 0,1,2,3 형태이므로 A,B,C,D로 변환
                            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                            self.test_prompts.append({
                                'prompt': prompt,
                                'answer': answer_map.get(sample['answer'], str(sample['answer'])),
                                'subject': subj,
                                'choices': {
                                    'A': sample['A'],
                                    'B': sample['B'],
                                    'C': sample['C'],
                                    'D': sample['D']
                                }
                            })
                print(f"MMLU에서 {len(self.test_prompts)}개 문제 로드 완료")
                
        elif self.benchmark_dataset == "mmlu":
            # MMLU (Massive Multitask Language Understanding) - 수학, 과학 등 57개 주제
            try:
                dataset = load_dataset("cais/mmlu", "all", split="test")
                # 각 주제에서 샘플 추출
                self.test_prompts = []
                subjects = [subject] if subject else data_subject
                
                for subj in subjects:
                    subject_data = [d for d in dataset if d['subject'] == subj]
                    if subject_data:
                        # data_count만큼 샘플 추출 (최대값 제한)
                        samples = random.sample(subject_data, min(data_count, len(subject_data)))
                        for sample in samples:
                            prompt = f"Question: {sample['question']}\n\nChoices:\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}\n\nPlease answer with just the letter (A, B, C, or D). Answer:"
                            # MMLU는 answer가 0,1,2,3 형태이므로 A,B,C,D로 변환
                            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                            self.test_prompts.append({
                                'prompt': prompt,
                                'answer': answer_map.get(sample['answer'], str(sample['answer'])),
                                'subject': subj,
                                'choices': {
                                    'A': sample['A'],
                                    'B': sample['B'],
                                    'C': sample['C'],
                                    'D': sample['D']
                                }
                            })
                print(f"MMLU에서 {len(self.test_prompts)}개 문제 로드 완료")
                
            except Exception as e:
                print(f"MMLU 로드 실패: {e}")
                self.use_default_prompts()
                
        elif self.benchmark_dataset == "gsm8k":
            # GSM8K - 초등학교 수학 문제
            try:
                dataset = load_dataset("gsm8k", "main", split="test")
                samples = random.sample(list(dataset), min(10, len(dataset)))
                self.test_prompts = []
                for sample in samples:
                    self.test_prompts.append({
                        'prompt': f"문제: {sample['question']}\n풀이:",
                        'answer': sample['answer'],
                        'subject': 'math'
                    })
                print(f"GSM8K에서 {len(self.test_prompts)}개 문제 로드 완료")
                
            except Exception as e:
                print(f"GSM8K 로드 실패: {e}")
                self.use_default_prompts()
                
        elif self.benchmark_dataset == "hellaswag":
            # HellaSwag - 상식 추론
            try:
                dataset = load_dataset("hellaswag", split="validation")
                samples = random.sample(list(dataset), min(10, len(dataset)))
                self.test_prompts = []
                for sample in samples:
                    prompt = f"{sample['ctx']}\n다음 중 가장 적절한 것은?\n"
                    for i, ending in enumerate(sample['endings']):
                        prompt += f"{i+1}) {ending}\n"
                    self.test_prompts.append({
                        'prompt': prompt,
                        'answer': str(sample['label']),
                        'subject': 'common_sense'
                    })
                print(f"HellaSwag에서 {len(self.test_prompts)}개 문제 로드 완료")
                
            except Exception as e:
                print(f"HellaSwag 로드 실패: {e}")
                self.use_default_prompts()
                
        elif self.benchmark_dataset == "arc":
            # ARC (AI2 Reasoning Challenge) - 과학 추론
            try:
                dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
                samples = random.sample(list(dataset), min(10, len(dataset)))
                self.test_prompts = []
                for sample in samples:
                    choices = sample['choices']
                    prompt = f"{sample['question']}\n"
                    for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
                        prompt += f"{label}) {text}\n"
                    self.test_prompts.append({
                        'prompt': prompt + "Answer:",
                        'answer': sample['answerKey'],
                        'subject': 'science'
                    })
                print(f"ARC에서 {len(self.test_prompts)}개 문제 로드 완료")
                
            except Exception as e:
                print(f"ARC 로드 실패: {e}")
                self.use_default_prompts()
                
        else:
            # 기본 프롬프트 사용
            self.use_default_prompts()
            
        
    def get_model_size(self, model) -> float:
        """모델 크기를 MB 단위로 계산 - 실제 파일 크기 측정"""
        import tempfile
        
        # 모델을 임시 파일로 저장하여 실제 크기 측정
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            file_size_bytes = os.path.getsize(tmp.name)
            file_size_mb = file_size_bytes / (1024 * 1024)
        
        return file_size_mb
        
    def apply_optimization(self, model, tokenizer, method: str):
        """선택한 최적화 방법을 모델에 적용"""
        print(f"\n{method} 최적화 적용 중...")
        
        if method == "baseline":
            # 최적화 없음
            return model, tokenizer
            
            
            
        elif method == "pruning":
            # PyTorch 구조적 프루닝
            print("구조적 프루닝 적용 중...")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # 20% 프루닝
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    # 프루닝을 영구적으로 적용
                    prune.remove(module, 'weight')
            print("20% 프루닝 적용 완료")
            return model, tokenizer
            
        elif method == "graph_optimize":
            # TorchScript를 이용한 그래프 최적화
            print("그래프 최적화 적용 중...")
            model.eval()
            try:
                # 모델의 forward 함수만 추적
                example_input = torch.randint(0, 1000, (1, 10))
                if torch.cuda.is_available():
                    example_input = example_input.cuda()
                    model = model.cuda()
                
                # TorchScript로 변환
                traced_model = torch.jit.trace(model, example_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                print("TorchScript 그래프 최적화 완료")
                return traced_model, tokenizer
            except Exception as e:
                print(f"그래프 최적화 실패: {e}")
                return model, tokenizer
                
        elif method == "torch_compile":
            # PyTorch 2.0+ torch.compile
            if hasattr(torch, 'compile'):
                print("torch.compile 최적화 적용 중...")
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile 최적화 완료")
            else:
                print("PyTorch 2.0+ 필요. torch.compile 건너뜀")
            return model, tokenizer
            
        elif method == "mixed_precision":
            # 자동 혼합 정밀도
            print("혼합 정밀도 (FP16) 적용 중...")
            model = model.half()  # FP16으로 변환
            if torch.cuda.is_available():
                model = model.cuda()
            print("혼합 정밀도 적용 완료")
            return model, tokenizer
            
        elif method == "dynamic_quantize":
            # PyTorch 동적 양자화 (더 많은 레이어 타입 포함)
            print("PyTorch 동적 양자화 적용 중...")
            model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            print("동적 양자화 적용 완료")
            return model, tokenizer
            
        elif method == "quanto_int8":
            # Quanto 8비트 양자화 (Mac 호환)
            try:
                from optimum.quanto import quantize, freeze, qint8
                print("Quanto 8비트 양자화 적용 중...")
                quantize(model, weights=qint8)
                freeze(model)
                print("Quanto 8비트 양자화 완료")
                return model, tokenizer
            except ImportError:
                print("Quanto가 설치되지 않았습니다. 설치: pip install optimum-quanto")
                return model, tokenizer
                
        elif method == "quanto_int4":
            # Quanto 4비트 양자화 (Mac 호환)
            try:
                from optimum.quanto import quantize, freeze, qint4
                print("Quanto 4비트 양자화 적용 중...")
                quantize(model, weights=qint4)
                freeze(model)
                print("Quanto 4비트 양자화 완료")
                return model, tokenizer
            except ImportError:
                print("Quanto가 설치되지 않았습니다. 설치: pip install optimum-quanto")
                return model, tokenizer
                
        else:
            print(f"알 수 없는 최적화 방법: {method}")
            return model, tokenizer
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 측정"""
        memory = {
            'cpu_percent': psutil.cpu_percent(),
            'cpu_memory_gb': psutil.Process().memory_info().rss / (1024**3),
            'gpu_memory_mb': 0,
            'gpu_percent': 0
        }
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                memory['gpu_memory_mb'] = gpus[0].memoryUsed
                memory['gpu_percent'] = gpus[0].memoryUtil * 100
                
        return memory
    
    def benchmark_inference(self, model, tokenizer, optimization_name: str) -> BenchmarkResult:
        """모델 추론 성능 벤치마크"""
        print(f"\n{optimization_name} 벤치마크 실행 중...")
        
        # 모델 크기 측정
        model_size = self.get_model_size(model)
        print(f"모델 크기: {model_size:.2f} MB")
        
        # 워밍업
        try:
            with torch.no_grad():
                _ = tokenizer("Hello", return_tensors="pt")
        except:
            pass
        
        inference_times = []
        predictions = []  # 예측 결과 저장
        
        # 메모리 사용량 측정
        memory_before = self.get_memory_usage()
        
        for i, test_item in enumerate(self.test_prompts):
            # 딕셔너리인 경우와 문자열인 경우 처리
            if isinstance(test_item, dict):
                prompt = test_item['prompt']
                answer = test_item.get('answer', None)
                subject = test_item.get('subject', 'general')
            else:
                prompt = test_item
                answer = None
                subject = 'general'
                
            print(f"  프롬프트 {i+1}/{len(self.test_prompts)} 처리 중... (주제: {subject})")
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                if torch.cuda.is_available() and hasattr(model, 'device'):
                    input_ids = input_ids.to(model.device)
                
                # 추론 시간 측정
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    if hasattr(model, 'generate'):
                        # Transformers 모델
                        output_ids = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50,  # 객관식 답변을 위해 충분한 토큰 수
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    else:
                        # TorchScript 모델
                        output_ids = model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
                
                # 출력 텍스트 디코딩
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_text = output_text[len(prompt):].strip()
                
                # 객관식 문제인 경우 간단한 평가
                if answer and answer in ['A', 'B', 'C', 'D', '0', '1', '2', '3']:
                    # 생성된 텍스트에서 답 추출
                    generated_upper = generated_text.upper().strip()
                    model_answer = None
                    
                    # 첫 번째 단일 문자 답변 찾기
                    for char in generated_upper:
                        if char in ['A', 'B', 'C', 'D']:
                            model_answer = char
                            break
                    
                    # 답변을 못 찾았으면 생성된 텍스트 전체에서 패턴 찾기
                    if not model_answer:
                        import re
                        # "답: A", "Answer: B", "정답은 C" 등의 패턴
                        patterns = [
                            r'(?:답|answer|정답|선택|답변)[\s:：은는이가]*([A-D])',
                            r'^([A-D])[)\.\s:]',
                            r'([A-D])(?:[)\.\s:]|번|입니다|가\s|이\s)',
                            r'\b([A-D])\b(?![-\w])',  # 단독 A-D (단어 경계)
                            r'(?:따라서|그러므로|결론적으로).*?([A-D])',
                        ]
                        # 원본 텍스트와 대문자 변환 텍스트 둘 다 검색
                        for text in [generated_text, generated_upper]:
                            for pattern in patterns:
                                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                                if match:
                                    model_answer = match.group(1).upper()
                                    break
                            if model_answer:
                                break
                    
                    # 정답 비교 - 객관식과 주관식 모두 평가
                    is_correct = False
                    if model_answer:
                        # 객관식 답변 평가
                        if answer.isdigit():
                            correct_letter = chr(ord('A') + int(answer))
                            is_correct = (model_answer == correct_letter)
                        else:
                            is_correct = (model_answer == answer)
                    
                    # 주관식 답변 평가 - BLEU/ROUGE 점수 사용
                    bleu_score = 0.0
                    rouge_score = 0.0
                    
                    if test_item.get('choices'):
                        # 모든 선택지와 비교하여 가장 높은 점수 찾기
                        best_match = {'choice': None, 'bleu': 0.0, 'rouge': 0.0}
                        
                        for choice_key, choice_text in test_item['choices'].items():
                            # BLEU 점수 계산
                            from nltk.translate.bleu_score import sentence_bleu
                            reference = choice_text.split()
                            candidate = generated_text.split()
                            try:
                                choice_bleu = sentence_bleu([reference], candidate, 
                                                          weights=(0.25, 0.25, 0.25, 0.25))
                            except:
                                choice_bleu = 0.0
                            
                            # ROUGE 점수 계산
                            from rouge_score import rouge_scorer
                            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
                            scores = scorer.score(choice_text, generated_text)
                            choice_rouge = scores['rougeL'].fmeasure
                            
                            # 최고 점수 업데이트
                            if choice_bleu + choice_rouge > best_match['bleu'] + best_match['rouge']:
                                best_match = {'choice': choice_key, 'bleu': choice_bleu, 'rouge': choice_rouge}
                        
                        # 정답과 비교
                        if not is_correct and best_match['choice']:
                            bleu_score = best_match['bleu']
                            rouge_score = best_match['rouge']
                            
                            # 가장 유사한 선택지가 정답인 경우
                            if best_match['choice'] == answer and (bleu_score > 0.5 or rouge_score > 0.6):
                                is_correct = True
                                model_answer = f"{answer} (주관식 - BLEU:{bleu_score:.2f}, ROUGE:{rouge_score:.2f})"
                            elif not model_answer:  # 객관식 답변을 못 찾은 경우
                                # 가장 유사한 선택지를 답으로 사용
                                model_answer = f"{best_match['choice']} (유사도 최고)"
                    
                    # 상세한 결과 출력
                    print(f"\n    질문: {prompt.split('### 질문:')[1].split('### 선택지:')[0].strip()[:]}...")
                    
                    # 선택지 출력 (test_item에 choices가 있는 경우)
                    if test_item.get('choices'):
                        print("    선택지:")
                        for choice_key, choice_text in test_item['choices'].items():
                            is_correct_choice = (choice_key == answer)
                            marker = "✓" if is_correct_choice else " "
                            print(f"      {marker} {choice_key}) {choice_text[:]}...")
                    
                    # 모델 응답과 평가 결과
                    print(f"\n    정답: {answer}")
                    print(f"    모델 답변: {model_answer if model_answer else 'None (객관식 답변 추출 실패)'}")
                    print(f"    전체 생성 텍스트: {generated_text[:]}")
                    print(f"    평가: {'✅ 정답' if is_correct else '❌ 오답'}")
                    print(f"-"*60+"\n\n\n")
                    
                    # BLEU/ROUGE 점수 (주관식 평가인 경우)
                    if bleu_score > 0 or rouge_score > 0:
                        print(f"    BLEU: {bleu_score:.3f}, ROUGE: {rouge_score:.3f}")
                    
                    predictions.append({
                        'prompt': prompt,  # 전체 프롬프트 저장
                        'subject': subject,
                        'true_answer': answer,
                        'model_answer': model_answer,
                        'generated_text': generated_text,  # 전체 생성 텍스트 저장
                        'is_correct': is_correct,
                        'bleu_score': bleu_score,
                        'rouge_score': rouge_score
                    })
                else:
                    # 일반 텍스트 생성
                    predictions.append({
                        'prompt': prompt,  # 전체 프롬프트 저장
                        'subject': subject,
                        'true_answer': answer,
                        'model_answer': None,
                        'generated_text': generated_text,  # 전체 생성 텍스트 저장
                        'is_correct': None
                    })
                
            except Exception as e:
                import traceback
                print(f"  추론 중 오류: {e}")
                print(f"  상세 에러: {traceback.format_exc()}")
                inference_times.append(1.0)  # 기본값
                predictions.append({
                    'prompt': prompt,  # 전체 프롬프트 저장
                    'subject': subject,
                    'true_answer': answer,
                    'model_answer': None,  # ERROR 대신 None 사용
                    'generated_text': f"Error: {str(e)}",
                    'is_correct': False
                })
        
        memory_after = self.get_memory_usage()
        
        # 평균 계산
        avg_inference_time = np.mean(inference_times) if inference_times else 1.0
        avg_tokens_per_second = 50 / avg_inference_time  # 50 new tokens (max_new_tokens=50)
        
        # 정확도 계산
        correct_count = sum(1 for p in predictions if p.get('is_correct', False))
        total_with_answers = sum(1 for p in predictions if p.get('true_answer') is not None)
        accuracy = correct_count / total_with_answers if total_with_answers > 0 else 0.0
        
        result = BenchmarkResult(
            model_name=self.model_id,
            optimization=optimization_name,
            load_time=0,
            inference_times=inference_times,
            memory_usage={
                'gpu_mb_used': memory_after['gpu_memory_mb'] - memory_before['gpu_memory_mb'],
                'cpu_gb_used': memory_after['cpu_memory_gb'] - memory_before['cpu_memory_gb']
            },
            tokens_per_second=avg_tokens_per_second,
            model_size_mb=model_size,
            accuracy=accuracy,
            predictions=predictions
        )
        
        print(f"  평균 추론 시간: {avg_inference_time:.3f}초")
        print(f"  초당 토큰 수: {avg_tokens_per_second:.2f}")
        print(f"  GPU 메모리 사용: {result.memory_usage['gpu_mb_used']:.2f} MB")
        print(f"  정확도: {accuracy:.1%} ({correct_count}/{total_with_answers})")
        
        return result
    
    def run_benchmark(self, selected_methods: List[str] = None, model_type="original"):
        """선택된 최적화 방법들을 순차적으로 적용하여 벤치마크 실행"""
        if selected_methods is None:
            selected_methods = SELECTED_OPTIMIZATIONS
            
        print(f"\nLLM 최적화 벤치마크 시작: {self.model_id}")
        print(f"모델 타입: {model_type}")
        print(f"순차적으로 적용할 최적화 방법: {' → '.join(selected_methods) if selected_methods else '없음'}")
        print("="*60)
        
        try:
            # 1. 모델 로드 및 초기 벤치마크
            model_label = model_type
            
            print(f"\n[{model_label}] 모델 로드 중...")
            start_time = time.time()
            
            # Load tokenizer with proper handling for all models
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                # Handle cases where tokenizer returns bool or other unexpected types
                if not hasattr(tokenizer, 'encode') or isinstance(tokenizer, bool):
                    print(f"특수 tokenizer 감지, LlamaTokenizer로 대체 시도...")
                    from transformers import LlamaTokenizer
                    tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            except Exception as e:
                print(f"Tokenizer 로드 실패: {e}, LlamaTokenizer 사용")
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            
            # Set pad token if needed
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else tokenizer.eos_token_id
            
            # MXFP4를 bfloat16으로 fallback하도록 설정
            # Mac(MPS)에서는 BFloat16 미지원, float32 사용
            if torch.backends.mps.is_available():
                dtype = torch.float32
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
                
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else 'cpu',  # GPU 있으면 자동 배치
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=False,  # 4bit 로드 비활성화
                load_in_8bit=False   # 8bit 로드 비활성화
            )
            
            # Mac/MPS에서 BFloat16 문제 해결: 모든 파라미터를 float32로 변환
            if torch.backends.mps.is_available() or dtype == torch.float32:
                print("📋 모델을 float32로 변환 중...")
                model = model.float()  # 모든 파라미터를 float32로 변환
                
            # Mac에서는 mps (Metal Performance Shaders) 사용
            # MPS 관련 에러 때문에 일단 비활성화
            if False and torch.backends.mps.is_available():
                model = model.to('mps')
                print("🎮 Metal GPU (MPS) 사용 중")
            elif torch.cuda.is_available():
                model = model.to('cuda')
                print("🎮 CUDA GPU 사용 중")
            else:
                model = model.to('cpu')
                print("💻 CPU 사용 중")
            
            load_time = time.time() - start_time
            print(f"로드 시간: {load_time:.2f}초")
            
            # 초기 벤치마크 실행 (모델 타입에 맞는 이름 사용)
            initial_label = model_type if not selected_methods else model_type
            baseline_result = self.benchmark_inference(model, tokenizer, initial_label)
            baseline_result.load_time = load_time
            self.results[initial_label] = baseline_result
            
            # 2. 선택된 최적화들을 순차적으로 적용
            current_optimization_name = model_type
            for i, method in enumerate(selected_methods):
                if method not in OPTIMIZATION_METHODS:
                    print(f"알 수 없는 최적화 방법: {method}")
                    continue
                
                # 이전 단계까지의 최적화 이름 생성
                if i == 0:
                    current_optimization_name = f"{model_type}:{method}"
                else:
                    current_optimization_name += f"+{method}"
                
                print(f"\n[{current_optimization_name}] 최적화 적용 중...")
                
                # 현재 모델에 추가 최적화 적용
                model, tokenizer = self.apply_optimization(model, tokenizer, method)
                
                # 벤치마크 실행
                result = self.benchmark_inference(model, tokenizer, current_optimization_name)
                result.load_time = 0  # 추가 최적화는 로드 시간 없음
                self.results[current_optimization_name] = result
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"벤치마크 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """벤치마크 결과 요약 출력"""
        if not self.results:
            print("출력할 결과가 없습니다.")
            return
            
        print("\n" + "="*110)
        print(f"📊 모델 성능 비교: {self.model_id.split('/')[-1]}")
        print("="*110)
        print(f"{'최적화 상태':<15} {'추론시간(초)':<12} {'토큰/초':<7} {'모델크기(MB)':<7} {'정확도':<7} {'속도 개선':<6} {'크기 감소':<6}")
        print("-"*110)
        
        # 첫 번째 결과를 기준으로 사용 (baseline, finetuned, quantized 중 하나)
        baseline_key = list(self.results.keys())[0]
        baseline = self.results.get(baseline_key)
        if not baseline:
            print("기준 결과가 없습니다.")
            return
            
        baseline_time = np.mean(baseline.inference_times)
        baseline_size = baseline.model_size_mb
        baseline_tokens = baseline.tokens_per_second
        baseline_accuracy = baseline.accuracy
        
        for method, result in self.results.items():
            avg_time = np.mean(result.inference_times) if result.inference_times else 0
            
            # 개선률 계산
            speedup = (baseline_time / avg_time) if avg_time > 0 else 0
            size_reduction = (1 - result.model_size_mb / baseline_size) * 100 if baseline_size > 0 else 0
            accuracy_diff = result.accuracy - baseline_accuracy
            
            # 색상 코드 (개선/악화)
            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{speedup:.2f}x ⚠️"
            size_str = f"{size_reduction:.1f}%" if size_reduction > 0 else f"{size_reduction:.1f}% ⚠️"
            accuracy_str = f"{result.accuracy:.1%}"
            if accuracy_diff > 0:
                accuracy_str += f" (+{accuracy_diff:.1%})"
            elif accuracy_diff < 0:
                accuracy_str += f" ({accuracy_diff:.1%}) ⚠️"
            
            print(f"{method:<25} {avg_time:<12.3f} {result.tokens_per_second:<10.2f} "
                  f"{result.model_size_mb:<12.2f} {accuracy_str:<10} {speedup_str:<12} {size_str:<12}")
        
        # 요약 통계
        print("\n" + "="*100)
        print("📈 요약")
        print("="*100)
        print(f"🔹 {baseline_key} 성능: {baseline_time:.3f}초/추론, {baseline_tokens:.2f} 토큰/초, {baseline_size:.2f} MB")
        
        best_speed = max(self.results.items(), key=lambda x: x[1].tokens_per_second)
        best_size = min(self.results.items(), key=lambda x: x[1].model_size_mb)
        best_accuracy = max(self.results.items(), key=lambda x: x[1].accuracy)
        
        print(f"🏆 최고 속도: {best_speed[0]} ({best_speed[1].tokens_per_second:.2f} 토큰/초)")
        print(f"🏆 최소 크기: {best_size[0]} ({best_size[1].model_size_mb:.2f} MB)")
        print(f"🏆 최고 정확도: {best_accuracy[0]} ({best_accuracy[1].accuracy:.1%})")
        
        # 누적 최적화 결과 강조
        if len(SELECTED_OPTIMIZATIONS) > 0:
            final_key = '+'.join(SELECTED_OPTIMIZATIONS)
            if final_key in self.results:
                final_result = self.results[final_key]
                final_speedup = (baseline_time / np.mean(final_result.inference_times)) if np.mean(final_result.inference_times) > 0 else 0
                final_size_reduction = (1 - final_result.model_size_mb / baseline_size) * 100 if baseline_size > 0 else 0
                
                print(f"\n💡 누적 최적화 최종 결과 ({final_key}):")
                print(f"   - 속도 향상: {final_speedup:.2f}x (Baseline 대비)")
                print(f"   - 크기 감소: {final_size_reduction:.1f}%")
                print(f"   - 정확도: {final_result.accuracy:.1%}")
        
        print("\n* PyTorch 내장 최적화 기능:")
        print("  - quantize_8bit: PyTorch quantization.quantize_dynamic()")
        print("  - pruning: torch.nn.utils.prune")
        print("  - graph_optimize: torch.jit.trace & optimize_for_inference")
        print("  - torch_compile: torch.compile() (PyTorch 2.0+)")
        print("  - mixed_precision: model.half() (FP16)")
        print("  - dynamic_quantize: quantize_dynamic with multiple layer types")
    
    def visualize_results(self, save_path='optimization_results.png'):
        """결과 시각화"""
        if not self.results:
            return
            
        methods = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        title = f'LLM 최적화 벤치마크 결과'
        if self.subject:
            title += f' - {self.subject}'
        fig.suptitle(title, fontsize=16)
        
        # 1. 추론 시간
        ax1 = axes[0, 0]
        times = [np.mean(self.results[m].inference_times) for m in methods]
        ax1.bar(methods, times)
        ax1.set_ylabel('평균 추론 시간 (초)')
        ax1.set_title('추론 속도')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 토큰/초
        ax2 = axes[0, 1]
        tokens = [self.results[m].tokens_per_second for m in methods]
        ax2.bar(methods, tokens)
        ax2.set_ylabel('초당 토큰 수')
        ax2.set_title('처리 속도')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 모델 크기
        ax3 = axes[1, 0]
        sizes = [self.results[m].model_size_mb for m in methods]
        ax3.bar(methods, sizes)
        ax3.set_ylabel('모델 크기 (MB)')
        ax3.set_title('모델 압축')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 속도 향상
        ax4 = axes[1, 1]
        baseline_time = np.mean(self.results.get('baseline', list(self.results.values())[0]).inference_times)
        speedups = [baseline_time / np.mean(self.results[m].inference_times) for m in methods]
        ax4.bar(methods, speedups)
        ax4.set_ylabel('속도 향상 (배)')
        ax4.set_title('Baseline 대비 성능')
        ax4.axhline(y=1, color='r', linestyle='--')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        #plt.show()  플랏 ㅌ


def run_single_model_benchmark(model_id, optimizations=[], model_type="original", result_dir=None):
    """단일 모델에 대한 벤치마크 실행"""
    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"benchmark_comparison_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n결과 저장 폴더: {result_dir}")
    
    # 전체 결과를 저장할 딕셔너리
    all_results = {}
    
    # 각 주제별로 벤치마크 실행
    for subject in data_subject:
        print(f"\n{'='*80}")
        print(f"🔬 주제별 벤치마크 시작: {subject}")
        print(f"{'='*80}")
        
        # 주제별 벤치마크 실행
        benchmark = LLMOptimizationBenchmark(model_id=model_id, benchmark_dataset="kmmlu", subject=subject)
        benchmark.run_benchmark(optimizations, model_type=model_type)
        
        # 결과 저장
        subject_results = {}
        for method, result in benchmark.results.items():
            subject_results[method] = {
                'inference_times': result.inference_times,
                'tokens_per_second': result.tokens_per_second,
                'model_size_mb': result.model_size_mb,
                'accuracy': result.accuracy,
                'memory_usage': result.memory_usage,
                'predictions': result.predictions
            }
        
        all_results[subject] = subject_results
        
        # 주제별 결과 파일 저장 (모델 타입 포함)
        subject_file = os.path.join(result_dir, f"benchmark_{model_type}_{subject}.json")
        with open(subject_file, 'w', encoding='utf-8') as f:
            json.dump(subject_results, f, indent=2, ensure_ascii=False)
        
        # 주제별 시각화 저장 (모델 타입 포함)
        viz_file = os.path.join(result_dir, f"benchmark_{model_type}_{subject}.png")
        benchmark.visualize_results(save_path=viz_file)
        
        # 주제별 요약 출력
        benchmark.print_summary()
    
    # 전체 결과 저장 (모델 타입 포함)
    all_results_file = os.path.join(result_dir, f"all_results_{model_type}.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 모든 벤치마크 완료! 결과가 '{result_dir}' 폴더에 저장되었습니다.")
    return result_dir, all_results

def run_comparison():
    """
    3가지 모델 비교 실행:
    1. 기존 모델 (baseline)
    2. 파인튜닝된 모델
    3. 파인튜닝된 모델을 경량화한 버전
    """
    results_summary = {}
    
    # 공통 결과 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common_result_dir = f"benchmark_comparison_{timestamp}"
    os.makedirs(common_result_dir, exist_ok=True)
    print(f"\n📁 공통 결과 폴더: {common_result_dir}")
    
    # 1. 기존 모델 벤치마크
    print("\n" + "="*80)
    print("1️⃣  기존 모델 벤치마크 시작")
    print("="*80)
    result_dir_base, results_base = run_single_model_benchmark(
        base_model_id, 
        optimizations=[], 
        model_type="original",
        result_dir=common_result_dir
    )
    results_summary['original'] = {
        'result_dir': result_dir_base,
        'model_id': base_model_id
    }
    
    # 2. 파인튜닝 모델 벤치마크
    print("\n" + "="*80)
    print("2️⃣  파인튜닝 모델 벤치마크 시작")
    print("="*80)
    result_dir_fine, results_fine = run_single_model_benchmark(
        fine_model_id, 
        optimizations=[],  # 파인튜닝 모델 자체만 평가
        model_type="finetuned",
        result_dir=common_result_dir
    )
    results_summary['finetuned'] = {
        'result_dir': result_dir_fine,
        'model_id': fine_model_id
    }
    
    # 3. 경량화된 파인튜닝 모델 벤치마크
    print("\n" + "="*80)
    print("3️⃣  경량화된 파인튜닝 모델 벤치마크 시작")
    print("="*80)
    result_dir_quant, results_quant = run_single_model_benchmark(
        fine_model_id, 
        optimizations=SELECTED_OPTIMIZATIONS, 
        model_type="lightweight",
        result_dir=common_result_dir
    )
    results_summary['lightweight'] = {
        'result_dir': result_dir_quant,
        'model_id': fine_model_id,
        'optimizations': SELECTED_OPTIMIZATIONS
    }
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 3개 모델 비교 결과 요약")
    print("="*80)
    
    # 각 모델의 주요 지표 출력
    print(f"\n{'모델 종류':<20} {'정확도':<10} {'추론시간(초)':<15} {'모델크기(MB)':<15}")
    print("-"*60)
    
    # 각 모델의 all_results 파일 읽기
    for model_type in ['original', 'finetuned', 'lightweight']:
        result_file = os.path.join(common_result_dir, f"all_results_{model_type}.json")
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 각 모델 타입에 맞는 결과 가져오기
                for subject, methods in data.items():
                    # 모델 타입에 따른 결과 키 확인
                    result_key = None
                    if model_type in methods:
                        result_key = model_type
                    elif model_type == 'lightweight':
                        # 경량화 모델은 적용된 최적화를 동적으로 찾음
                        # 마지막으로 적용된 최적화 결과를 사용
                        for key in methods.keys():
                            if key.startswith('lightweight'):
                                result_key = key  # 계속 업데이트하여 마지막 결과 사용
                                break
                    
                    if result_key:
                        result = methods[result_key]
                        model_name = {
                            'original': '기존 모델',
                            'finetuned': '파인튜닝 모델',
                            'lightweight': '경량화 모델'
                        }[model_type]
                        
                        accuracy = result.get('accuracy', 0) * 100
                        avg_time = np.mean(result.get('inference_times', [1.0]))
                        model_size = result.get('model_size_mb', 0)
                        
                        print(f"{model_name:<20} {accuracy:<10.1f} {avg_time:<15.3f} {model_size:<15.2f}")
                        break
    
    print("\n📁 모든 결과가 저장된 폴더: " + common_result_dir)
    
    return results_summary

if __name__ == "__main__":
    # 비교 실행 옵션
    print("실행 옵션:")
    print("1. 3개 모델 비교 (기존 vs 파인튜닝 vs 경량화)")
    print("2. 단일 모델 벤치마크")
    
    choice = input("\n선택 (1 또는 2): ")
    
    if choice == "1":
        # 3개 모델 비교 실행
        run_comparison()
    else:
        # 기존 단일 모델 벤치마크
        print("\n사용 가능한 최적화 방법:")
        for i, method in enumerate(OPTIMIZATION_METHODS):
            print(f"{i}: {method}")
        
        # 결과 저장 폴더 생성 (모델이름_날짜_시간)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_id.split('/')[-1]
        result_dir = f"{model_name}_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        print(f"\n결과 저장 폴더: {result_dir}")
    
        # 전체 결과를 저장할 딕셔너리
        all_results = {}
        
        # 각 주제별로 벤치마크 실행
        for subject in data_subject:
            print(f"\n{'='*80}")
            print(f"🔬 주제별 벤치마크 시작: {subject}")
            print(f"{'='*80}")
            
            # 주제별 벤치마크 실행
            benchmark = LLMOptimizationBenchmark(model_id=model_id, benchmark_dataset="kmmlu", subject=subject)
            benchmark.run_benchmark(SELECTED_OPTIMIZATIONS)
            
            # 결과 저장
            subject_results = {}
            for method, result in benchmark.results.items():
                subject_results[method] = {
                    'inference_times': result.inference_times,
                    'tokens_per_second': result.tokens_per_second,
                    'model_size_mb': result.model_size_mb,
                    'accuracy': result.accuracy,
                    'memory_usage': result.memory_usage,
                    'predictions': result.predictions
                }
            
            all_results[subject] = subject_results
            
            # 주제별 결과 파일 저장
            subject_file = os.path.join(result_dir, f"benchmark_{subject}.json")
            with open(subject_file, 'w', encoding='utf-8') as f:
                json.dump(subject_results, f, indent=2, ensure_ascii=False)
            
            # 주제별 시각화 저장
            viz_file = os.path.join(result_dir, f"benchmark_{subject}.png")
            benchmark.visualize_results(save_path=viz_file)
            
            # 주제별 요약 출력
            benchmark.print_summary()
        
        # 전체 결과 저장
        all_results_file = os.path.join(result_dir, "all_results.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 모든 벤치마크 완료! 결과가 '{result_dir}' 폴더에 저장되었습니다.")