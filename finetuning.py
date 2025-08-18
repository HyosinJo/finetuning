import os
import sys

#리워드모델(답변품질) , 하이브리드 방식
#GRPO 데이터셋으로 변경 KMMLU
#파인튜닝 데이터셋 선정 리워드모델선정

# 리워드모델 변경
# 테스트 

# lora 어댑터 moe에서 위치 확인
# 학습 에포크,스탭나오게
# 벤치마크 나오게
# 모델 질문 응답 나오게

# 로컬 TRL 경로를 Python path에 추가
TRL_PATH = os.path.join(os.path.dirname(__file__), 'trl_finetune')
if os.path.exists(TRL_PATH):
    sys.path.insert(0, TRL_PATH)
    print(f"✅ 로컬 TRL 사용: {TRL_PATH}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
import numpy as np
from datetime import datetime
import json
import random
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# W&B 비활성화
os.environ["WANDB_DISABLED"] = "true"

# Hugging Face 토큰 설정
HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)  
        print("✅ Hugging Face 로그인 성공")
    except Exception as e:
        print(f"⚠️ Hugging Face 로그인 실패: {e}")
        print("환경변수 HF_TOKEN을 설정하거나 huggingface-cli login을 실행하세요.")
else:
    print("⚠️ HF_TOKEN이 설정되지 않았습니다. .env 파일에 HF_TOKEN을 추가하세요.")
    HF_TOKEN = None

# 설정 변수

MODEL_ID = 'google/gemma-3-270m'
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ID = "facebook/MobileLLM-600M"
MODEL_ID = "trillionlabs/Tri-7B"
MODEL_ID = "openai/gpt-oss-20b"


FINE_TUNE_FRAMEWORK = "trl"  # "trl" 또는 "verl (DAPO 인경우)" 선택
METHOD = "GRPO"  # "SFT", "DPO", "GRPO", "PPO", "DAPO", "ORPO" 중 선택

# 데이터셋 설정
DATASET_TYPE = 'heegyu/CoT-collection-ko' #COT데이터셋
DATASET_TYPE = "KMMLU"  
KMMLU_SUBJECT = "Economics"

# 현재 선택된 데이터셋 (방법에 따라 자동 선택됨)
if DATASET_TYPE == "KMMLU":
    # KMMLU 사용 시 모든 방법에 KMMLU 데이터 사용
    DATASET_NAME = "HAERAE-HUB/KMMLU"
else:
    # 기존 로직 유지
    if METHOD == "SFT":
        DATASET_NAME = "markrAI/KoCommercial-Dataset" 
    elif METHOD == "DPO" or METHOD == "ORPO":
        DATASET_NAME = "maywell/ko_Ultrafeedback_binarized"  # DPO와 ORPO 둘 다 선호/비선호 쌍 필요
    elif METHOD == "DAPO":
        DATASET_NAME = "markrAI/KoCommercial-Dataset"  # DAPO는 정답이 있는 데이터 필요
    else:  # PPO, GRPO
        DATASET_NAME = "kyujinpy/KOR-OpenOrca-Platypus-v3"



# LoRA/QLoRA 설정
USE_LORA = True  # True: LoRA 사용, False: 풀 파인튜닝
USE_QLORA = False  # True: QLoRA (4bit), False: 일반 LoRA 또는 풀 파인튜닝 - macOS는 QLoRA 미지원
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout



# STF 학습 설정
num_train_epochs=100 # 에포크
SAVE_STEPS = 100  # 몇 스텝마다 저장할지
MAX_STEPS = 5000  # 총 학습 스텝
DATA_SIZE = 100  # 사용할 데이터 개수 (파인튜닝에 적합한 크기 필요)
LEARNING_RATE = 5e-4  # LoRA는 더 높은 학습률 사용
 # GRPO의 GRPO_num_generations = 10 로 나누어떨어지도록 수정
BATCH_SIZE = 10
OUTPUT_DIR = f"finetune_{MODEL_ID.split('/')[-1]}_{METHOD}_{'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else 'Full'}_{DATASET_TYPE}"

# GRPO 학습 설정
num_batch_iteration=10
SAVE_STEPS = 30  # 몇 스텝마다 저장할지
MAX_STEPS = 5000  # 총 학습 스텝
DATA_SIZE = 100  # 사용할 데이터 개수 (파인튜닝에 적합한 크기 필요)
LEARNING_RATE = 5e-4
GRPO_num_generation = 10 #oi
BATCH_SIZE = 10
OUTPUT_DIR = f"finetune_{MODEL_ID.split('/')[-1]}_{METHOD}_{'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else 'Full'}_{DATASET_TYPE}"


# 각 학습 방법별 최적화된 한국어 데이터셋:
# SFT: markrAI/KoCommercial-Dataset (상업용 한국어 instruction 데이터셋)
# DPO: maywell/ko_Ultrafeedback_binarized (선호/비선호 쌍이 있는 한국어 데이터셋)
# PPO/GRPO: kyujinpy/KOR-OpenOrca-Platypus-v3 (고품질 한국어 instruction)


# KMMLU 데이터 준비 함수들
def prepare_kmmlu_for_sft():
    """SFT용 KMMLU 데이터 준비 - 문제와 정답"""
    all_data = []
    
    print(f"\n📊 KMMLU 데이터 로드 중 - 주제: {KMMLU_SUBJECT}")
    
    try:
        # 전체 데이터셋 크기 확인
        train_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="train")
        dev_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="dev")
        test_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="test")
        
        print(f"📈 {KMMLU_SUBJECT} 전체 데이터 통계:")
        print(f"   - Train: {len(train_full)}개")
        print(f"   - Dev: {len(dev_full)}개")
        print(f"   - Test: {len(test_full)}개")
        print(f"   - 총합: {len(train_full) + len(dev_full) + len(test_full)}개")
        
        # 특정 주제에서만 데이터 가져오기
        dataset = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split=f"train[:{DATA_SIZE}]")
        
        for item in dataset:
            all_data.append({
                'question': item['question'],
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'answer': item['answer'],
                'subject': KMMLU_SUBJECT
            })
        
        print(f"✅ {KMMLU_SUBJECT}에서 {len(all_data)}개 데이터 로드 완료")
        
    except Exception as e:
        print(f"⚠️ {KMMLU_SUBJECT} 로드 실패: {e}")
        raise
    
    # datasets.Dataset으로 변환
    from datasets import Dataset
    dataset = Dataset.from_list(all_data)
    
    def preprocess(examples):
        texts = []
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            choices = [examples['A'][i], examples['B'][i], examples['C'][i], examples['D'][i]]
            answer_idx = examples['answer'][i]
            # KMMLU는 1-indexed (1,2,3,4)이므로 검증
            if answer_idx < 1 or answer_idx > 4:
                print(f"경고: 잘못된 answer 인덱스 {answer_idx}, 1로 설정")
                answer_idx = 1
            answer_letter = ['A', 'B', 'C', 'D'][answer_idx]
            
            text = f"문제: {question}\n"
            text += f"A) {choices[0]}\n"
            text += f"B) {choices[1]}\n" 
            text += f"C) {choices[2]}\n"
            text += f"D) {choices[3]}\n"
            text += f"정답: {answer_letter}"
            texts.append(text)
        return {"text": texts}
    
    processed_dataset = dataset.map(preprocess, batched=True)
    
    # 데이터 샘플 출력
    print("\n📋 SFT 데이터 샘플:")
    print("="*80)
    for i in range(min(3, len(processed_dataset))):
        print(f"\n[샘플 {i+1}]")
        print(processed_dataset[i]['text'])
    print("="*80)
    
    return processed_dataset

def prepare_kmmlu_for_dpo():
    """DPO용 KMMLU 데이터 준비 - 선호/비선호 쌍"""
    all_data = []
    
    print(f"\n📊 KMMLU DPO 데이터 로드 중 - 주제: {KMMLU_SUBJECT}")
    
    try:
        # 전체 데이터셋 크기 확인
        train_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="train")
        dev_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="dev")
        test_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="test")
        
        print(f"📈 {KMMLU_SUBJECT} 전체 데이터 통계:")
        print(f"   - Train: {len(train_full)}개")
        print(f"   - Dev: {len(dev_full)}개")
        print(f"   - Test: {len(test_full)}개")
        print(f"   - 총합: {len(train_full) + len(dev_full) + len(test_full)}개")
        
        # 특정 주제에서만 데이터 가져오기
        dataset = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split=f"train[:{DATA_SIZE}]")
        
        for item in dataset:
            question = item['question']
            choices = [item['A'], item['B'], item['C'], item['D']]
            correct_idx = item['answer']
            
            # answer 인덱스 범위 확인
            if correct_idx < 0 or correct_idx > 3:
                print(f"경고: 잘못된 answer 인덱스 {correct_idx}, 0으로 설정")
                correct_idx = 0
            
            prompt = f"문제: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n답:"
            
            # 정답을 chosen, 오답 중 하나를 rejected로
            chosen = ['A', 'B', 'C', 'D'][correct_idx]
            wrong_indices = [i for i in range(4) if i != correct_idx]
            rejected_idx = random.choice(wrong_indices)
            rejected = ['A', 'B', 'C', 'D'][rejected_idx]
            
            all_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "subject": KMMLU_SUBJECT
            })
        
        print(f"✅ {KMMLU_SUBJECT}에서 {len(all_data)}개 DPO 데이터 로드 완료")
        
    except Exception as e:
        print(f"⚠️ {KMMLU_SUBJECT} 로드 실패: {e}")
        raise
    
    from datasets import Dataset
    dataset = Dataset.from_list(all_data)
    
    # 데이터 샘플 출력
    print("\n📋 DPO 데이터 샘플:")
    print("="*80)
    for i in range(min(3, len(all_data))):
        print(f"\n[샘플 {i+1}]")
        print(f"프롬프트: {all_data[i]['prompt']}")
        print(f"선호 응답 (정답): {all_data[i]['chosen']}")
        print(f"비선호 응답 (오답): {all_data[i]['rejected']}")
        print(f"주제: {all_data[i]['subject']}")
    print("="*80)
    
    return dataset

def prepare_kmmlu_for_rlhf():
    """ KMMLU 데이터 준비 - 프롬프트만"""
    all_data = []
    
    print(f"\n📊 KMMLU RLHF 데이터 로드 중 - 주제: {KMMLU_SUBJECT}")
    
    try:
        # 전체 데이터셋 크기 확인
        train_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="train")
        dev_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="dev")
        test_full = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split="test")
        
        print(f"📈 {KMMLU_SUBJECT} 전체 데이터 통계:")
        print(f"   - Train: {len(train_full)}개")
        print(f"   - Dev: {len(dev_full)}개")
        print(f"   - Test: {len(test_full)}개")
        print(f"   - 총합: {len(train_full) + len(dev_full) + len(test_full)}개")
        
        # 특정 주제에서만 데이터 가져오기
        dataset = load_dataset("HAERAE-HUB/KMMLU", KMMLU_SUBJECT, split=f"train[:{DATA_SIZE}]")
        
        for item in dataset:
            all_data.append({
                'question': item['question'],
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'answer': item['answer'],
                'subject': KMMLU_SUBJECT
            })
        
        print(f"✅ {KMMLU_SUBJECT}에서 {len(all_data)}개 RLHF 데이터 로드 완료")
        
    except Exception as e:
        print(f"⚠️ {KMMLU_SUBJECT} 로드 실패: {e}")
        raise
    
    # datasets.Dataset으로 변환
    from datasets import Dataset
    dataset = Dataset.from_list(all_data)
    
    def preprocess(examples):
        prompts = []
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            choices = [examples['A'][i], examples['B'][i], examples['C'][i], examples['D'][i]]
            
            prompt = f"문제: {question}\n"
            prompt += f"A) {choices[0]}\n"
            prompt += f"B) {choices[1]}\n"
            prompt += f"C) {choices[2]}\n"
            prompt += f"D) {choices[3]}\n"
            prompt += "정답을 고르세요:"
            prompts.append(prompt)
        return {"prompt": prompts, "answer": examples['answer']}
    
    processed_dataset = dataset.map(preprocess, batched=True)
    
    # 데이터 샘플 출력
    print("\n📋 RLHF(PPO/GRPO) 데이터 샘플:")
    print("="*80)
    for i in range(min(3, len(processed_dataset))):
        print(f"\n[샘플 {i+1}]")
        print("프롬프트:")
        print("-" * 60)
        print(processed_dataset[i]['prompt'])
        print("-" * 60)
        answer_idx = processed_dataset[i]['answer']
        # KMMLU는 1-indexed (1,2,3,4)이므로 0-indexed로 변환
        answer_letter = ['A', 'B', 'C', 'D'][answer_idx - 1]
        print(f"정답: {answer_letter} (원본 인덱스: {answer_idx})")
    print("="*80)
    
    return processed_dataset





# 한국어 지시 따르기 데이터 준비 함수들
def prepare_korean_instruction_for_sft():
    """SFT용 한국어 지시 따르기 데이터 준비"""
    # markrAI/KoCommercial-Dataset 사용
    dataset = load_dataset("markrAI/KoCommercial-Dataset", split=f"train[:{DATA_SIZE}]")
    
    def preprocess(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i] if examples['input'][i] else ""
            output = examples['output'][i]
            
            # 지시-입력-응답 형식으로 포맷팅
            if input_text:
                text = f"### 지시:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n{output}"
            else:
                text = f"### 지시:\n{instruction}\n\n### 응답:\n{output}"
            texts.append(text)
        return {"text": texts}
    
    return dataset.map(preprocess, batched=True)

def prepare_korean_instruction_for_dpo():
    """DPO용 한국어 지시 따르기 데이터 준비 - 선호/비선호 쌍 생성"""
    # maywell/ko_Ultrafeedback_binarized 사용 (선호/비선호 쌍이 있는 데이터셋)
    dataset = load_dataset("maywell/ko_Ultrafeedback_binarized", split=f"train[:{DATA_SIZE}]")
    
    def preprocess(examples):
        prompts = []
        chosens = []
        rejecteds = []
        
        for i in range(len(examples['prompt'])):
            # 이미 prompt, chosen, rejected 형식으로 되어 있음
            prompt = examples['prompt'][i]
            chosen = examples['chosen'][i][-1]['content'] if isinstance(examples['chosen'][i], list) else examples['chosen'][i]
            rejected = examples['rejected'][i][-1]['content'] if isinstance(examples['rejected'][i], list) else examples['rejected'][i]
            
            # 프롬프트 포맷팅
            formatted_prompt = f"### 지시:\n{prompt}\n\n### 응답:\n"
            
            prompts.append(formatted_prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)
        
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
    
    return dataset.map(preprocess, batched=True)



def prepare_korean_instruction_for_rlhf():
    """RLHF(PPO/GRPO)용 한국어 지시 따르기 데이터 준비"""
    # kyujinpy/KOR-OpenOrca-Platypus-v3 사용
    dataset = load_dataset("kyujinpy/KOR-OpenOrca-Platypus-v3", split=f"train[:{DATA_SIZE}]")
    
    def preprocess(examples):
        prompts = []
        reference_outputs = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i] if examples['input'][i] else ""
            output = examples['output'][i]
            
            # 프롬프트 포맷팅
            if input_text:
                prompt = f"### 지시:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n"
            else:
                prompt = f"### 지시:\n{instruction}\n\n### 응답:\n"
                
            prompts.append(prompt)
            reference_outputs.append(output)
        
        return {"prompt": prompts, "reference_output": reference_outputs}
    
    return dataset.map(preprocess, batched=True)

def prepare_korean_instruction_for_dapo():
    """DAPO용 한국어 지시 따르기 데이터 준비 - Parquet 파일 생성"""
    import pandas as pd
    
    # markrAI/KoCommercial-Dataset 사용 (정답이 있는 instruction 데이터)
    dataset = load_dataset("markrAI/KoCommercial-Dataset", split=f"train[:{DATA_SIZE}]")
    
    dapo_data = []
    
    for idx, item in enumerate(dataset):
        # DAPO 형식으로 변환
        data_item = {
            "data_source": "KoCommercial-Dataset",
            "prompt": [{"role": "user", "content": item['instruction'] + (f"\n{item['input']}" if item['input'] else "")}],
            "ability": "korean_instruction",
            "reward_model": {
                "style": "rule",
                "ground_truth": item['output']  # 정답으로 사용
            },
            "extra_info": {
                "split": "train" if idx < int(DATA_SIZE * 0.8) else "val",
                "index": idx,
                "original_instruction": item['instruction'],
                "original_input": item['input'],
                "original_output": item['output']
            }
        }
        dapo_data.append(data_item)
    
    # DataFrame으로 변환
    df = pd.DataFrame(dapo_data)
    
    # Train/Val 분할 (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # 절대 경로로 파일 저장
    output_dir_abs = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir_abs, exist_ok=True)
    
    train_file = os.path.join(output_dir_abs, "dapo_train.parquet")
    val_file = os.path.join(output_dir_abs, "dapo_val.parquet")
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    
    print(f"✅ DAPO 학습 데이터: {len(train_df)}개 - {train_file}")
    print(f"✅ DAPO 검증 데이터: {len(val_df)}개 - {val_file}")
    
    # 샘플 데이터 출력
    if len(dapo_data) > 0:
        print("\n📋 DAPO 데이터 샘플:")
        print("="*80)
        sample = dapo_data[0]
        print(f"프롬프트: {sample['prompt'][0]['content'][:200]}...")
        print(f"정답: {sample['reward_model']['ground_truth'][:200]}...")
        print("="*80)
    
    return train_file, val_file




# SFT 방식 파인튜닝 (Supervised Fine-Tuning)
def train_sft():
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    
    print(f"🚀 SFT 파인튜닝 시작... ({'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else '풀 파인튜닝'})")
    
    # QLoRA를 위한 양자화 설정
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if USE_QLORA else None,
        torch_dtype=torch.float32 if not USE_QLORA else None,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # QLoRA 준비
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)
    
    # Mac/MPS에서 BFloat16 문제 해결
    if torch.backends.mps.is_available():
        print("📋 Mac 환경 감지 - 모델을 float32로 변환 중...")
        model = model.float()
    
    # LoRA 설정
    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if "MobileLLM" in MODEL_ID:
        # MobileLLM은 LlamaTokenizer 직접 사용
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        except Exception as e:
            if "custom_code" in str(e) or "trust_remote_code" in str(e):
                print("⚠️ 이 모델은 custom code가 필요합니다. trust_remote_code=True로 재시도...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
            else:
                raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터셋 준비 (SFT용)
    if DATASET_TYPE == "KMMLU":
        dataset = prepare_kmmlu_for_sft()
    else:
        dataset = prepare_korean_instruction_for_sft()
    
    # SFT 설정
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        save_steps=SAVE_STEPS,
        push_to_hub=False,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # MPS에서 pin_memory 비활성화
        report_to="tensorboard",  # 텐서보드 로그
        logging_dir="./logs",     
    )
    
    # SFT 트레이너
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # 학습
    trainer.train()
    
    # 최종 모델 저장
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# DPO 방식 파인튜닝  
def train_dpo():
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    
    print(f"🚀 DPO 파인튜닝 시작... ({'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else '풀 파인튜닝'})")
    
    # QLoRA를 위한 양자화 설정
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if USE_QLORA else None,
        torch_dtype=torch.float32 if not USE_QLORA else None,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # 참조 모델은 LoRA/QLoRA 사용 시 None (자동 처리됨)
    ref_model = None
    if not (USE_LORA or USE_QLORA):
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
    
    # QLoRA 준비
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)
    
    # Mac/MPS에서 BFloat16 문제 해결
    if torch.backends.mps.is_available():
        print("📋 Mac 환경 감지 - 모델을 float32로 변환 중...")
        model = model.float()
    
    # LoRA 설정
    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if "MobileLLM" in MODEL_ID:
        # MobileLLM은 LlamaTokenizer 직접 사용
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        except Exception as e:
            if "custom_code" in str(e) or "trust_remote_code" in str(e):
                print("⚠️ 이 모델은 custom code가 필요합니다. trust_remote_code=True로 재시도...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
            else:
                raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # DPO 설정
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=num_train_epochs,
        save_steps=SAVE_STEPS,
        logging_steps=1,
        beta=0.1,  # KL 페널티
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # MPS에서 pin_memory 비활성화
    )
    
    # 데이터셋 준비 (DPO용)
    if DATASET_TYPE == "KMMLU":
        train_dataset = prepare_kmmlu_for_dpo()
    else:
        train_dataset = prepare_korean_instruction_for_dpo()
    
    # DPO 트레이너
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # 학습
    dpo_trainer.train()
    
    # 최종 모델 저장
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")





# ORPO 방식 파인튜닝
def train_orpo():
    from trl import ORPOTrainer, ORPOConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    
    print(f"🚀 ORPO 파인튜닝 시작... ({'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else '풀 파인튜닝'})")
    
    # QLoRA를 위한 양자화 설정
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if USE_QLORA else None,
        torch_dtype=torch.float32 if not USE_QLORA else None,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # QLoRA 준비
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)
    
    # Mac/MPS에서 BFloat16 문제 해결
    if torch.backends.mps.is_available():
        print("📋 Mac 환경 감지 - 모델을 float32로 변환 중...")
        model = model.float()
    
    # LoRA 설정
    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if "MobileLLM" in MODEL_ID:
        # MobileLLM은 LlamaTokenizer 직접 사용
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        except Exception as e:
            if "custom_code" in str(e) or "trust_remote_code" in str(e):
                print("⚠️ 이 모델은 custom code가 필요합니다. trust_remote_code=True로 재시도...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
            else:
                raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ORPO 설정
    orpo_config = ORPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=num_train_epochs,
        save_steps=SAVE_STEPS,
        logging_steps=1,
        beta=0.1,  # SFT loss 가중치
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # MPS에서 pin_memory 비활성화
    )
    
    # 데이터셋 준비 (ORPO용 - DPO와 동일한 형식)
    if DATASET_TYPE == "KMMLU":
        train_dataset = prepare_kmmlu_for_dpo()
    else:
        train_dataset = prepare_korean_instruction_for_dpo()
    
    # ORPO 트레이너
    orpo_trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # 학습
    orpo_trainer.train()
    
    # 최종 모델 저장
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# GRPO 방식 파인튜닝 (TRL 내장)
def train_grpo():
    from trl import GRPOTrainer, GRPOConfig
    print("🚀 GRPO 파인튜닝 시작 (TRL 내장)...")
    
    # 토크나이저 로드
    if "MobileLLM" in MODEL_ID:
        # MobileLLM은 LlamaTokenizer 직접 사용
        print("🦙 MobileLLM 모델 감지 - LlamaTokenizer 사용")
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        except Exception as e:
            if "custom_code" in str(e) or "trust_remote_code" in str(e):
                print("⚠️ 이 모델은 custom code가 필요합니다. trust_remote_code=True로 재시도...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
            else:
                raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # token_type_ids 제거 설정
    # token_type_ids는 BERT 계열 모델에서 문장 구분을 위해 사용되는 파라미터
    # GPT/Llama 계열 모델(Qwen, DeepSeek 등)에서는 사용하지 않음
    # 하지만 일부 토크나이저가 이를 자동으로 생성해서 generate() 함수에서 오류 발생
    # 따라서 model_input_names에서 제거하여 토크나이저가 이를 생성하지 않도록 함
    if hasattr(tokenizer, 'model_input_names') and 'token_type_ids' in tokenizer.model_input_names:
        tokenizer.model_input_names.remove('token_type_ids')
    
    # 데이터셋 준비
    if DATASET_TYPE == "KMMLU":
        dataset = prepare_kmmlu_for_rlhf()
    else:
        dataset = prepare_korean_instruction_for_rlhf()
    
    # 보상 모델 초기화 (한 번만 로드)
    print("💯 보상 모델 로드 중...")
    try:
        from transformers import AutoModelForSequenceClassification
        # 다국어 지원 보상 모델 사용
        #reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        reward_model_name = "gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B" # rm r1 추론 보상 모델
        reward_model_name = "heegyu/ko-reward-model-1.3b-v0.1" # 한글 리워드 모델
        reward_model_name = "heegyu/ko-reward-model-safety-1.3b-v0.2" # 한글리워드 2 
        reward_model_name = "heegyu/ko-reward-model-helpful-1.3b-v0.2" # 한글리워드 3 유용한 답변에 점수

        # MPS에서는 CPU로 로드
        if torch.backends.mps.is_available():
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        else:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        print(f"✅ 보상 모델 로드 성공: {reward_model_name}")
    except Exception as e:
        print(f"⚠️ 보상 모델 로드 실패: {e}")
        print("임베딩 기반 보상으로 대체합니다.")
        reward_model = None
        
        # 대체: 한국어 임베딩 모델
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('jhgan/ko-sbert-sts')
            print("✅ 한국어 임베딩 모델 로드 성공")
        except:
            embedder = None
            print("⚠️ 임베딩 모델도 실패. 기본 규칙 사용")
    
    # KMMLU 리워드 함수 정의
    def reward_func(completions, prompts=None, **kwargs):
        """KMMLU 답변에 대한 리워드 계산 - 보상 모델 기반"""
        rewards = []
        
        # kwargs에서 정답 정보 추출
        if 'answer' in kwargs:
            answers = kwargs['answer']
            if not isinstance(answers, list):
                answers = [answers] * len(completions)
        else:
            return [0.0 for _ in completions]
        
        # 프롬프트 정보 (없으면 기본값)
        if prompts is None:
            prompts = [""] * len(completions)
        elif not isinstance(prompts, list):
            prompts = [prompts] * len(completions)
        
        # 실시간 로그 출력 헤더 (첫 번째 배치만)
        if len(completions) > 0:
            print("\n" + "="*100)
            print("🔍 보상 계산 시작 - 배치 크기:", len(completions))
            print("="*100)
        
        for i, (completion, answer_idx) in enumerate(zip(completions, answers)):
            completion_text = completion.strip()
            # 보상 모델이 있는 경우
            if reward_model is not None:
                print('reward model 보상 평가 시작')
                try:
                    # 보상모델이 평가할 품질 데이터 (질문지,모델의 응답)
                    full_text = f"질문 : {prompts[i]}\n 모델 응답 : {completion_text}"
                    
                    # 보상 모델로 품질 평가
                    inputs = reward_tokenizer(
                        full_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    )
                    
                    # 보상 모델의 디바이스로 이동
                    if hasattr(reward_model, 'device'):
                        device = reward_model.device
                    else:
                        # 모델의 첫 번째 파라미터의 디바이스 확인
                        device = next(reward_model.parameters()).device
                    
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = reward_model(**inputs)
                        # logits 차원 처리
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                            if logits.dim() == 2:  # [batch_size, num_classes]
                                quality_score = logits[0, 0].item()  # 첫 번째 클래스 점수
                            elif logits.dim() == 1:  # [num_classes]
                                quality_score = logits[0].item()
                            else:
                                quality_score = logits.item()  # 스칼라
                        else:
                            quality_score = outputs[0].item() if hasattr(outputs[0], 'item') else float(outputs[0])
                    
                    # 정답 여부 확인 (A,B,C,D 추출)
                    import re
                    extracted_answer = None
                    patterns = [
                        r'^([A-D])[).\s]?',
                        r'(?:답|정답|선택)(?:은|는)?\s*[:\s]?\s*([A-D])',
                        r'([A-D])\s*(?:번|입니다|이다|임)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, completion_text.upper())
                        if match:
                            extracted_answer = match.group(1)
                            break
                    
                    # 최종 보상: 품질 점수 + 정답 보너스
                    # KMMLU는 1-indexed (1,2,3,4)이므로 0-indexed로 변환
                    correct_answer = ['A', 'B', 'C', 'D'][answer_idx - 1]
                    
                    if extracted_answer == correct_answer:
                        # 정답: 품질 점수 + 보너스
                        reward = quality_score + 1
                        is_correct = "✅ 정답"
                    elif extracted_answer:
                        # 답은했지만 틀린경우: 품질 점수 - 페널티
                        reward = quality_score - 1
                        is_correct = f"❌ 오답 (선택: {extracted_answer}, 정답: {correct_answer})"
                    else:
                        # 품질도 안좋고 답도 틀린경우: 추가 패널티
                        reward = quality_score - 1.5
                        is_correct = f"⚠️ 응답 품질 저하 (정답: {correct_answer})"
                    
                    # 실시간 로그 출력
                    print(f"\n[샘플 {i+1}]")
                    print(f"📝 문제: {prompts[i][:100]}...")
                    print(f"💬 응답: {completion_text[:200]}...")
                    print(f"📊 보상모델의 점수: {quality_score:.4f}")
                    print(f"🎯 정답 체크: {is_correct}")
                    print(f"🏆 최종 보상: {reward:.4f}")
                    print("-" * 80)
                    
                except Exception as e:
                    print(f"보상 계산 오류: {e}")
                    reward = 0.0
                    
            # 임베딩 모델이 있는 경우 (폴백)
            elif 'embedder' in locals() and embedder is not None:
                try:
                    # 정답 선택지 텍스트 가져오기 (실제로는 dataset에서 가져와야 함)
                    correct_idx = answer_idx
                    # KMMLU는 1-indexed (1,2,3,4)이므로 0-indexed로 변환
                    correct_letter = ['A', 'B', 'C', 'D'][correct_idx - 1]
                    
                    # 응답과 각 선택지의 유사도 계산
                    response_emb = embedder.encode([completion_text])
                    
                    # 엄격한 답안 추출
                    import re
                    extracted_answer = None
                    patterns = [
                        r'^([A-D])[).\s]?',
                        r'(?:답|정답|선택)(?:은|는)?\s*[:\s]?\s*([A-D])',
                        r'([A-D])\s*(?:번|입니다|이다|임)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, completion_text.upper())
                        if match:
                            extracted_answer = match.group(1)
                            break
                    
                    # 정답 체크
                    if extracted_answer == correct_letter:
                        reward = 1.0
                        is_correct = f"✅ 정답 (임베딩 기반)"
                    elif extracted_answer:
                        reward = -0.5
                        is_correct = f"❌ 오답 (임베딩 기반, 선택: {extracted_answer}, 정답: {correct_letter})"
                    else:
                        reward = -1.0
                        is_correct = f"⚠️ 형식 오류 (임베딩 기반, 정답: {correct_letter})"
                    
                    # 실시간 로그 출력
                    print(f"\n[샘플 {i+1}]")
                    print(f"📝 문제: {prompts[i]}")
                    print(f"💬 응답: {completion_text[:]}")
                    print(f"🎯 정답 체크: {is_correct}")
                    print(f"🏆 최종 보상: {reward:.4f}")
                    print("-" * 80)
                        
                except Exception as e:
                    print(f"임베딩 계산 오류: {e}")
                    reward = 0.0
                    
            else:
                # 기본 규칙 기반 (폴백의 폴백)
                import re
                # KMMLU는 1-indexed (1,2,3,4)이므로 0-indexed로 변환
                correct_answer = ['A', 'B', 'C', 'D'][answer_idx - 1]
                
                # 엄격한 답안 추출 패턴
                extracted_answer = None
                patterns = [
                    r'^([A-D])[).\s]?',  # 문장 시작 부분의 A), A., A 등
                    r'(?:답|정답|선택)(?:은|는)?\s*[:\s]?\s*([A-D])',  # 답은 A, 정답: B 등
                    r'([A-D])\s*(?:번|입니다|이다|임)',  # A번, A입니다 등
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, completion_text.upper())
                    if match:
                        extracted_answer = match.group(1)
                        break
                
                # 정답 체크
                if extracted_answer == correct_answer:
                    reward = 1.0
                    is_correct = f"✅ 정답 (규칙 기반)"
                elif extracted_answer:
                    reward = -1.0
                    is_correct = f"❌ 오답 (규칙 기반, 선택: {extracted_answer}, 정답: {correct_answer})"
                else:
                    reward = -1.5
                    is_correct = f"⚠️ 형식 오류 (규칙 기반, 정답: {correct_answer})"
                
                # 실시간 로그 출력
                print(f"\n[샘플 {i+1}]")
                print(f"📝 문제: {prompts[i][:100]}...")
                print(f"💬 응답: {completion_text[:200]}...")
                print(f"🎯 정답 체크: {is_correct}")
                print(f"🏆 최종 보상: {reward:.4f}")
                print("-" * 80)
            
            rewards.append(reward)
        
        return rewards
    
    # GPU 감지 및 DeepSpeed 자동 설정 (엘리스에서는 비활성화)
    deepspeed_config = None
    if torch.cuda.is_available() and False:  # 엘리스에서는 DeepSpeed 비활성화
        print("🚀 GPU 감지됨! DeepSpeed 자동 활성화")
        
        # GPU 메모리에 따라 ZeRO stage 자동 선택
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB 단위
        
        if gpu_memory < 8:  # 8GB 미만
            zero_stage = 3  # 최대 메모리 절약
            offload_optimizer = True
            offload_param = True
            print(f"💾 GPU 메모리 {gpu_memory:.1f}GB - ZeRO Stage 3 + CPU 오프로딩 사용")
        elif gpu_memory < 16:  # 16GB 미만
            zero_stage = 2
            offload_optimizer = True
            offload_param = False
            print(f"💾 GPU 메모리 {gpu_memory:.1f}GB - ZeRO Stage 2 + 옵티마이저 오프로딩 사용")
        else:  # 16GB 이상
            zero_stage = 1
            offload_optimizer = False
            offload_param = False
            print(f"💾 GPU 메모리 {gpu_memory:.1f}GB - ZeRO Stage 1 사용")
        
        # bf16과 fp16 중 하나만 선택
        use_bf16 = torch.cuda.is_bf16_supported()
        
        deepspeed_config = {
            "train_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": not use_bf16,  # bf16이 지원되지 않을 때만 fp16 사용
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": use_bf16  # bf16이 지원되면 bf16 사용
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if offload_optimizer else "none",
                    "pin_memory": True
                } if offload_optimizer else {},
                "offload_param": {
                    "device": "cpu" if offload_param else "none",
                    "pin_memory": True
                } if offload_param else {},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        
        # DeepSpeed 설정 파일 저장
        import json
        with open("ds_config.json", "w") as f:
            json.dump(deepspeed_config, f, indent=2)
        print("✅ DeepSpeed 설정 파일 저장됨: ds_config.json")
    
    # GRPO 설정
    grpo_config = GRPOConfig(
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_iterations=num_batch_iteration,  # 각 배치당 반복 횟수
        epsilon=0.2,  # surr loss 클리핑 값
        save_steps=SAVE_STEPS,
        output_dir=OUTPUT_DIR,
        max_completion_length=128,  # max_new_tokens 대신 max_completion_length 사용 (응답 외대)
        max_prompt_length=1024,  # 프롬프트 최대 길이 (질문 최대)
        num_generations=GRPO_num_generation,  # 각 프롬프트당 생성할 응답 수
        temperature=1.0,  # 생성 온도
        beta=0.1,  # KL 페널티 계수
        logging_steps=1, 
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,  # GPU가 지원하면 자동 활성화
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # bf16 미지원시 fp16 사용
        report_to="tensorboard",  # 텐서보드 로그
        logging_dir="./logs",
        deepspeed=deepspeed_config,  # DeepSpeed 설정 추가 (GPU 있을 때만)
        importance_sampling_level="token",  # 중요도 샘플링 수준: "token" 또는 "sequence"
        scale_rewards=True,  # 보상 정규화 여부 (표준편차로 나누기)
        use_liger_loss=False,  # Liger 커널 사용 여부 (GPU 최적화, token-level만 지원)
    )
    
    # Trajectory 저장을 위한 콜백 클래스
    from transformers import TrainerCallback
    import pickle
    
    class TrajectoryCallback(TrainerCallback):
        def __init__(self):
            self.trajectories = []
            
        def on_step_end(self, args, state, control, **kwargs):
            # 10스텝마다 GPU 메모리 사용량 출력
            if state.global_step % 10 == 0:
                print_gpu_memory()
            
            # 매 스텝마다 trajectory 수집
            # state.log_history에서 최신 로그 가져오기
            if state.log_history:
                latest_log = state.log_history[-1]
                
                trajectory = {
                    'step': state.global_step,
                    'epoch': state.epoch,
                    
                    # 핵심 메트릭
                    'loss': latest_log.get('loss', None),
                    'reward': latest_log.get('reward', None),
                    'reward_mean': latest_log.get('rewards/reward_func/mean', None),
                    'reward_std': latest_log.get('rewards/reward_func/std', None),
                    
                    # KL divergence (정책 변화 측정)
                    'kl': latest_log.get('kl', None),
                    'entropy': latest_log.get('entropy', None),
                    
                    # PPO 클리핑 비율 (정책 업데이트 크기)
                    'clip_ratio_mean': latest_log.get('clip_ratio/region_mean', None),
                    'clip_ratio_low': latest_log.get('clip_ratio/low_mean', None),
                    'clip_ratio_high': latest_log.get('clip_ratio/high_mean', None),
                    
                    # 그래디언트 정보
                    'grad_norm': latest_log.get('grad_norm', None),
                    'learning_rate': latest_log.get('learning_rate', None),
                    
                    # 생성 정보
                    'completions_mean_length': latest_log.get('completions/mean_length', None),
                    'completions_clipped_ratio': latest_log.get('completions/clipped_ratio', None),
                    
                    # 전체 로그 (필요시)
                    'full_log': latest_log
                }
                
                self.trajectories.append(trajectory)
            
        def on_save(self, args, state, control, **kwargs):
            # 체크포인트 저장 시 trajectory도 저장
            save_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            traj_path = os.path.join(save_path, "trajectories.pkl")
            
            # 디렉토리 생성
            os.makedirs(save_path, exist_ok=True)
            
            # Trajectory 저장
            with open(traj_path, 'wb') as f:
                pickle.dump(self.trajectories, f)
            print(f"💾 Trajectory 저장됨: {traj_path}")
            
            # JSON 형식으로도 저장 (읽기 쉽게)
            import json
            json_path = os.path.join(save_path, "trajectories.json")
            # pickle 불가능한 객체를 문자열로 변환
            json_trajectories = []
            for traj in self.trajectories:
                # 전체 로그 제외하고 모든 필드 복사 (full_log는 너무 크므로 제외)
                json_traj = {k: v for k, v in traj.items() if k != 'full_log'}
                # float 변환이 필요한 경우만 처리
                if json_traj.get('kl') is not None:
                    json_traj['kl'] = float(json_traj['kl'])
                json_trajectories.append(json_traj)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_trajectories, f, indent=2, ensure_ascii=False)
    
    # GPU 메모리 모니터링 함수
    def print_gpu_memory():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"\n🎮 GPU {i} ({torch.cuda.get_device_name(i)}) 메모리 사용량:")
                print(f"   할당됨: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"   예약됨: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                print(f"   전체: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"   사용률: {(torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100:.1f}%")
    
    # 모델 로드 (MobileLLM 등 커스텀 모델 지원)
    print(f"🤖 모델 로드 중... ({'QLoRA' if USE_QLORA else 'LoRA' if USE_LORA else '풀 파인튜닝'})")
    
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # QLoRA 설정
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    
    if "MobileLLM" in MODEL_ID:
        # MobileLLM은 trust_remote_code 필요
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=bnb_config if USE_QLORA else None,
            torch_dtype=torch.float32 if not USE_QLORA else None,
            device_map="auto"
        )
    else:
        # Config 먼저 로드하여 quantization_config 문제 해결
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # quantization_config가 None이면 빈 dict로 설정
        if hasattr(config, 'quantization_config') and config.quantization_config is None:
            print("⚠️ quantization_config가 None입니다. 빈 dict로 설정...")
            config.quantization_config = {}
        
        # 수정된 config로 모델 로드
        # 환경에 따라 다른 설정 사용
        if torch.backends.mps.is_available():
            # Mac 환경
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,
                trust_remote_code=True,
                config=config,
                quantization_config=bnb_config if USE_QLORA else None,
                torch_dtype=torch.float32,  # Mac에서는 항상 float32
                device_map="cpu",  # CPU 명시적 지정
                low_cpu_mem_usage=True  # 메모리 효율적 로딩
            )
        else:
            # GPU 환경 (엘리스 등)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,
                trust_remote_code=True,
                config=config,
                quantization_config=bnb_config if USE_QLORA else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
    
    # QLoRA 준비
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)
    
    # Mac/MPS에서 BFloat16 문제 해결
    if torch.backends.mps.is_available():
        print("📋 Mac 환경 감지 - 모델을 float32로 변환 중...")
        model = model.float()
    
    # LoRA 설정
    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Mac/MPS에서 BFloat16 문제 해결
    if torch.backends.mps.is_available():
        print("📋 Mac 환경 감지 - 모델을 float32로 변환 중...")
        model = model.float()
    
    # 모델 로드 후 GPU 메모리 사용량 출력
    print("\n📊 모델 로드 완료!")
    print_gpu_memory()
    
    # 생성 전 디버깅
    print("\n🔍 테스트 생성 시작...")
    test_prompt = "문제: 한국의 수도는? A) 서울 B) 부산 C) 대구 D) 인천\n정답을 고르세요:"
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    print(f"입력 토큰 수: {test_inputs['input_ids'].shape}")
    
    with torch.no_grad():
        test_output = model.generate(
            **test_inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.7
        )
    print(f"생성된 텍스트: {tokenizer.decode(test_output[0], skip_special_tokens=True)}")
    print("✅ 테스트 생성 완료\n")
    
    # GRPO 트레이너
    trainer = GRPOTrainer(
        model=model,  # 모델 객체 직접 전달
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,  # 단일 리워드 함수도 reward_funcs 파라미터 사용
        processing_class=tokenizer,  # 실제 토크나이저 객체 전달
    )
    
    # 콜백 추가
    trajectory_callback = TrajectoryCallback()
    trainer.add_callback(trajectory_callback)
    
    # 학습
    trainer.train()
    
    # 최종 trajectory 저장
    final_traj_path = os.path.join(OUTPUT_DIR, "final", "trajectories.pkl")
    os.makedirs(os.path.join(OUTPUT_DIR, "final"), exist_ok=True)
    with open(final_traj_path, 'wb') as f:
        pickle.dump(trajectory_callback.trajectories, f)
    
    # 최종 모델 저장
    trainer.save_model(f"{OUTPUT_DIR}/final")

# DAPO 방식 파인튜닝 (VERL 사용)
def train_dapo():
    if FINE_TUNE_FRAMEWORK == "verl":
        print("🚀 DAPO 파인튜닝 시작 (VERL 내장)...")
        
        # 먼저 DAPO용 데이터셋 준비
        print("\n📋 DAPO용 한국어 데이터셋 준비 중...")
        train_file, val_file = prepare_korean_instruction_for_dapo()
        
        import subprocess
        import sys
        
        # VERL 경로 설정
        verl_path = "/Users/ai/llm_proj/verl_finetune/verl"
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{verl_path}:{env.get('PYTHONPATH', '')}"
        env["WANDB_MODE"] = "offline"  # WandB 오프라인 모드로 설정
        
        # GPU 설정 확인
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device = "cuda"
        elif torch.backends.mps.is_available():
            n_gpus = 0  # Mac MPS는 GPU로 카운트하지 않음
            device = "cpu"  # VERL은 MPS를 직접 지원하지 않으므로 CPU 사용
        else:
            n_gpus = 0
            device = "cpu"
        
        print(f"🖥️  감지된 디바이스: {device}, GPU 수: {n_gpus}")
        
        # DAPO를 모듈로 실행 (-m 옵션 사용)
        cmd = [
            sys.executable,
            "-m",
            "recipe.dapo.main_dapo",
            f"actor_rollout_ref.model.path={MODEL_ID}",
            f"trainer.total_epochs=1",
            f"data.train_batch_size={BATCH_SIZE}",
            f"actor_rollout_ref.actor.optim.lr={LEARNING_RATE}",
            f"trainer.default_local_dir={OUTPUT_DIR}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size={BATCH_SIZE}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={BATCH_SIZE}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size={BATCH_SIZE}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size={BATCH_SIZE}",
            f"critic.ppo_micro_batch_size_per_gpu={BATCH_SIZE}",
            f"critic.ppo_mini_batch_size={BATCH_SIZE}",
            f"data.train_files={train_file}",  # 실제 학습 데이터
            f"data.val_files={val_file}",      # 실제 검증 데이터
            f"trainer.n_gpus_per_node={n_gpus}",  # GPU 수 설정
            f"trainer.nnodes=1",  # 단일 노드
            f"trainer.device={device}",  # 디바이스 설정
            "trainer.logger=['console']",  # WandB 대신 콘솔만 사용
            "data.max_prompt_length=2048",  # 프롬프트 최대 길이 더 늘림
            "data.truncation=right",  # 긴 프롬프트는 오른쪽 자르기
        ]
        
        print(f"실행 명령: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, env=env, cwd=verl_path, check=True)
            print("✅ DAPO 파인튜닝 완료!")
        except subprocess.CalledProcessError as e:
            print(f"❌ DAPO 실행 오류: {e}")
            return
        
        # 모델과 토크나이저 로드
        if "MobileLLM" in MODEL_ID:
            # MobileLLM은 LlamaTokenizer 직접 사용
            tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
            except Exception as e:
                if "custom_code" in str(e) or "trust_remote_code" in str(e):
                    print("⚠️ 이 모델은 custom code가 필요합니다. trust_remote_code=True로 재시도...")
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
                else:
                    raise e
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 데이터셋 준비
        if DATASET_TYPE == "KMMLU":
            dataset = prepare_kmmlu_for_rlhf()
        else:
            dataset = prepare_korean_instruction_for_rlhf()
            

    else:
        print("⚠️  DAPO는 VERL 프레임워크에서만 사용 가능합니다.")
        print("FINE_TUNE_FRAMEWORK를 'verl'로 설정하세요.")

# TRL을 사용한 파인튜닝
def train_with_trl():
    try:
        import trl
    except ImportError:
        print("TRL이 설치되어 있지 않습니다. 설치 중...")
        os.system("pip install trl")
    
    # METHOD에 따라 적절한 학습 함수 호출
    if METHOD.upper() == "SFT":
        train_sft()
    elif METHOD.upper() == "DPO":
        train_dpo()
    elif METHOD.upper() == "ORPO":
        train_orpo()
    elif METHOD.upper() == "GRPO":
        train_grpo()
    elif METHOD.upper() == "DAPO":
        print("⚠️  DAPO는 VERL에서만 지원됩니다.")
        print("FINE_TUNE_FRAMEWORK를 'verl'로 설정하세요.")
    else:
        print(f"❌ 지원하지 않는 방법: {METHOD}")
        print("TRL 지원 방법: SFT, PPO, DPO, ORPO, GRPO")

# VERL을 사용한 파인튜닝
def train_with_verl():
    
    import subprocess
    import sys

    # VERL 디렉토리 경로
    verl_dir = "/Users/ai/llm_proj/verl_finetune/verl"
    
    # VERL 리포지토리 확인
    if not os.path.exists(verl_dir):
        print("⚠️ VERL 리포지토리가 없습니다. 클론 중...")
        os.makedirs("/Users/ai/llm_proj/verl_finetune", exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/volcengine/verl.git", verl_dir], check=True)
    
    # verl_finetune을 Python 경로에 추가
    sys.path.insert(0, "/Users/ai/llm_proj/verl_finetune")

    import verl_finetune.verl.verl as verl
    print(f"✅ VERL {verl.__version__} 설치 확인")

    if METHOD.upper() == "DAPO":
        train_dapo()
    else:
        print(f"⚠️  VERL은 현재 DAPO만 지원합니다.")
        print("다른 방법은 TRL을 사용하세요.")

# 메인 실행
def main():
    print(f"""
================================================================================
🎯 LLM 파인튜닝 스크립트
================================================================================
모델: {MODEL_ID}
프레임워크: {FINE_TUNE_FRAMEWORK}
메소드: {METHOD}
파인튜닝 타입: {'QLoRA (4bit)' if USE_QLORA else 'LoRA' if USE_LORA else '풀 파인튜닝'}
데이터셋: {DATASET_TYPE} ({KMMLU_SUBJECT if DATASET_TYPE == "KMMLU" else ""})
데이터 크기: {DATA_SIZE}
학습률: {LEARNING_RATE}
배치 크기: {BATCH_SIZE}
저장 간격: {SAVE_STEPS} 스텝
총 스텝: {MAX_STEPS}
출력 디렉토리: {OUTPUT_DIR}
================================================================================
""")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 설정 저장
    config = {
        "model_id": MODEL_ID,
        "framework": FINE_TUNE_FRAMEWORK,
        "method": METHOD,
        "save_steps": SAVE_STEPS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 프레임워크에 따라 학습 실행
    if FINE_TUNE_FRAMEWORK.lower() == "trl":
        train_with_trl()
    elif FINE_TUNE_FRAMEWORK.lower() == "verl":
        train_with_verl()
    else:
        print(f"❌ 지원하지 않는 프레임워크: {FINE_TUNE_FRAMEWORK}")
        print("'trl' 또는 'verl'을 선택하세요.")

if __name__ == "__main__":
    main()