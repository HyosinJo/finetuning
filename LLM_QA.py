#!/usr/bin/env python3
"""
LLM Q&A 인터페이스
직접 질문하고 모델이 응답하는 대화형 스크립트
"""

import os
import time
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# ==================== 설정 ====================
# HuggingFace 모델 또는 로컬 체크포인트 경로

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # HuggingFace 원본 모델
MODEL_ID = "facebook/MobileLLM-600M"
MODEL_ID = 'trillionlabs/Tri-7B'
# MODEL_ID = "/Users/ai/llm_proj/finetune_DeepSeek-R1-Distill-Qwen-1.5B_GRPO_LoRA_KMMLU/checkpoint-150"  # 로컬 체크포인트
#MODEL_ID = "/Users/ai/llm_proj/finetune_DeepSeek-R1-Distill-Qwen-1.5B_GRPO_LoRA_KMMLU/checkpoint-150"


MODEL_ID = "/Users/ai/llm_proj/finetune_gpt-oss-20b_SFT_LoRA_heegyu/CoT-collection-ko/checkpoint-1890"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
# ============================================

class LLMQA:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"\n🤖 LLM Q&A 시스템 시작")
        print(f"📍 모델 경로: {model_id}")
        print(f"💻 디바이스: {self.device}")
        print("="*60)
        
        # 모델과 토크나이저 로드
        self.load_model()
        
    def load_model(self):
        """모델과 토크나이저 로드"""
        print("\n⏳ 모델 로딩 중...")
        start_time = time.time()
        
        try:
            # 토크나이저 로드 (MobileLLM 특수 처리)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True
                )
                if isinstance(self.tokenizer, bool):
                    print("🦙 MobileLLM 토크나이저 감지 - LlamaTokenizer 사용")
                    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            except:
                print("🦙 토크나이저 로드 실패 - LlamaTokenizer 사용")
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            
            # pad token 설정
            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 - 로컬 경로인지 HuggingFace 모델인지 확인
            is_local = os.path.exists(self.model_id)
            adapter_config_path = os.path.join(self.model_id, "adapter_config.json") if is_local else None
            
            if is_local and adapter_config_path and os.path.exists(adapter_config_path):
                # LoRA 어댑터 로드
                print("📎 LoRA 어댑터 감지")
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_id = adapter_config.get("base_model_name_or_path", "facebook/MobileLLM-600M")
                
                # 베이스 모델 로드
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map='auto' if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN")
                )
                
                # LoRA 어댑터 적용
                self.model = PeftModel.from_pretrained(base_model, self.model_id)
                self.model = self.model.merge_and_unload()  # 병합하여 속도 향상
            else:
                # 일반 모델 로드 (HuggingFace 또는 로컬 전체 모델)
                print(f"🤗 {'HuggingFace' if not is_local else '로컬'} 모델 로드 중...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map='auto' if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN") if not is_local else None
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            model_size = self.get_model_size()
            
            print(f"✅ 모델 로드 완료!")
            print(f"⏱️  로드 시간: {load_time:.2f}초")
            print(f"📦 모델 크기: {model_size:.2f} MB")
            print(f"🔧 파라미터 수: {self.count_parameters():,}")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def get_model_size(self):
        """모델 크기 계산 (MB)"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    def count_parameters(self):
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.model.parameters())
    
    def generate_response(self, question):
        """질문에 대한 응답 생성"""
        # 프롬프트 포맷팅 - 간단한 대화형으로 변경
        prompt = f"Human: {question}\n\nAssistant:"
        
        # 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # token_type_ids가 있으면 제거 (일부 모델은 지원하지 않음)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        # 모델의 dtype으로 입력 텐서 변환
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(self.device).to(model_dtype) if v.dtype.is_floating_point else v.to(self.device) 
                  for k, v in inputs.items()}
        
        # 추론 시간 측정
        start_time = time.time()
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 응답 부분만 추출
        if "### 응답:" in generated_text:
            response = generated_text.split("### 응답:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        # 토큰 수 계산
        num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = num_tokens / inference_time if inference_time > 0 else 0
        
        return response, inference_time, tokens_per_second, num_tokens
    
    def interactive_qa(self):
        """대화형 Q&A 세션"""
        print("\n" + "="*60)
        print("💬 대화형 Q&A 시작! (종료: 'quit', 'exit', 또는 Ctrl+C)")
        print("="*60)
        
        while True:
            try:
                # 사용자 입력
                print("\n" + "🧑 사용자: ", end="")
                question = input().strip()
                
                # 종료 조건
                if question.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 Q&A 세션을 종료합니다.")
                    break
                
                if not question:
                    continue
                
                # 응답 생성
                print("\n⚡ 생성 중...", end="\r")
                response, inference_time, tokens_per_second, num_tokens = self.generate_response(question)
                
                # 결과 출력
                print(f"\n🤖 모델: {response}")
                print(f"\n📊 벤치마크:")
                print(f"   ⏱️  추론 시간: {inference_time:.3f}초")
                print(f"   ⚡ 속도: {tokens_per_second:.1f} 토큰/초")
                print(f"   📝 생성 토큰: {num_tokens}개")
                print(f"   💾 메모리 사용: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
                
            except KeyboardInterrupt:
                print("\n\n👋 Q&A 세션을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                continue

def main():
    """메인 함수"""
    print("🚀 LLM Q&A 시스템 v1.0")
    print("="*60)
    
    # Q&A 시스템 초기화
    qa_system = LLMQA(MODEL_ID)
    
    # 대화형 세션 시작
    qa_system.interactive_qa()
    
    print("\n✅ 프로그램 종료")

if __name__ == "__main__":
    main()