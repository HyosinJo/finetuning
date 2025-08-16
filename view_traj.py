#!/usr/bin/env python3
"""
Trajectory Visualization Tool
Usage: python view_traj.py [trajectory_json_path]
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import subprocess
from datetime import datetime


traj_path =  "/Users/ai/llm_proj/finetune_DeepSeek-R1-Distill-Qwen-1.5B_GRPO_LoRA_KMMLU/checkpoint-150/trajectories.json"





def load_trajectory(json_path):
    """JSON 파일에서 trajectory 데이터 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_trajectories(data, save_dir):
    """trajectory 데이터를 그래프로 시각화"""
    # 데이터 추출
    steps = [d['step'] for d in data]
    
    # 메트릭별 데이터 수집
    metrics = {
        'loss': [],
        'reward': [],
        'kl': [],
        'entropy': [],
        'learning_rate': [],
        'grad_norm': [],
        'clip_ratio_mean': []
    }
    
    for d in data:
        for key in metrics:
            if key in d and d[key] is not None:
                metrics[key].append(d[key])
            else:
                metrics[key].append(np.nan)
    
    # 서브플롯 생성
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Loss 플롯
    # Loss: 모델이 얼마나 잘 학습하고 있는지 나타내는 핵심 지표
    # 낮을수록 좋음. 일반적으로 학습이 진행되면서 감소해야 함
    ax = axes[0]
    mask = ~np.isnan(metrics['loss'])
    ax.plot(np.array(steps)[mask], np.array(metrics['loss'])[mask], 'b-', marker='o', markersize=3)
    ax.set_title('Loss over Steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Reward 플롯 (10스텝마다만 있음)
    # Reward: GRPO에서 모델 응답의 품질을 나타내는 지표
    # - 품질 점수(보통 음수) + 정답 보너스/페널티
    # - 높을수록 좋음. GRPO는 10스텝마다 데이터 수집 후 보상 계산
    ax = axes[1]
    reward_steps = []
    reward_values = []
    for i, (s, r) in enumerate(zip(steps, metrics['reward'])):
        if not np.isnan(r):
            reward_steps.append(s)
            reward_values.append(r)
    if reward_steps:
        ax.plot(reward_steps, reward_values, 'r-', marker='s', markersize=8, linewidth=2)
        ax.scatter(reward_steps, reward_values, c='red', s=100, zorder=5)
        # 보상값 텍스트 추가
        for s, r in zip(reward_steps, reward_values):
            ax.annotate(f'{r:.3f}', (s, r), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax.set_title('Reward over Steps (every 10 steps)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    
    # 3. KL Divergence 플롯
    # KL Divergence: 현재 모델이 원본 모델에서 얼마나 벗어났는지 측정
    # - 너무 높으면: 모델이 원본과 너무 달라짐 (과적합 위험)
    # - 너무 낮으면: 학습이 충분히 안됨
    # - 일반적으로 0.01~0.1 범위가 적당
    ax = axes[2]
    mask = ~np.isnan(metrics['kl'])
    ax.semilogy(np.array(steps)[mask], np.array(metrics['kl'])[mask], 'g-', marker='o', markersize=3)
    ax.set_title('KL Divergence over Steps (log scale)')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Divergence')
    ax.grid(True, alpha=0.3)
    
    # 4. Entropy 플롯
    # Entropy: 모델 출력의 불확실성/다양성 측정
    # - 높으면: 모델이 여러 옵션을 고려 (탐색적)
    # - 낮으면: 모델이 특정 답변에 확신 (결정적)
    # - 너무 낮으면 다양성 부족, 너무 높으면 불안정
    ax = axes[3]
    mask = ~np.isnan(metrics['entropy'])
    ax.plot(np.array(steps)[mask], np.array(metrics['entropy'])[mask], 'm-', marker='o', markersize=3)
    ax.set_title('Entropy over Steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.grid(True, alpha=0.3)
    
    # 5. Learning Rate 플롯
    # Learning Rate: 학습 속도를 제어하는 하이퍼파라미터
    # - 높으면: 빠른 학습 but 불안정할 수 있음
    # - 낮으면: 안정적 but 느린 학습
    # - 보통 스케줄러에 의해 점진적으로 감소
    ax = axes[4]
    mask = ~np.isnan(metrics['learning_rate'])
    ax.plot(np.array(steps)[mask], np.array(metrics['learning_rate'])[mask], 'c-', marker='o', markersize=3)
    ax.set_title('Learning Rate over Steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 6. Gradient Norm 플롯 (log scale)
    # Gradient Norm: 그래디언트의 크기 (학습 신호의 강도)
    # - 너무 크면: gradient exploding (발산) 위험
    # - 너무 작으면: vanishing gradient (학습 정체)
    # - 안정적인 학습에서는 일정한 범위 유지
    ax = axes[5]
    mask = ~np.isnan(metrics['grad_norm'])
    ax.semilogy(np.array(steps)[mask], np.array(metrics['grad_norm'])[mask], 'orange', marker='o', markersize=3)
    ax.set_title('Gradient Norm over Steps (log scale)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.grid(True, alpha=0.3)
    
    # 7. Clip Ratio 플롯
    # Clip Ratio: PPO/GRPO에서 업데이트가 클리핑된 비율
    # - 0에 가까우면: 업데이트가 너무 작음 (학습 부족)
    # - 1에 가까우면: 업데이트가 너무 큼 (불안정)
    # - 0.1~0.3 정도가 이상적
    ax = axes[6]
    mask = ~np.isnan(metrics['clip_ratio_mean'])
    ax.plot(np.array(steps)[mask], np.array(metrics['clip_ratio_mean'])[mask], 'brown', marker='o', markersize=3)
    ax.set_title('Clip Ratio Mean over Steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Clip Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 8. Loss vs Reward Scatter (보상이 있는 스텝만)
    # Loss와 Reward의 상관관계:
    # - 이상적으로는 음의 상관관계 (Loss↓, Reward↑)
    # - 패턴이 없으면 학습이 제대로 안됨
    # - 양의 상관관계면 뭔가 잘못됨
    ax = axes[7]
    loss_at_reward = []
    reward_only = []
    for i, (l, r) in enumerate(zip(metrics['loss'], metrics['reward'])):
        if not np.isnan(r):
            loss_at_reward.append(l)
            reward_only.append(r)
    if loss_at_reward:
        ax.scatter(loss_at_reward, reward_only, c='purple', s=100, alpha=0.6)
        ax.set_title('Loss vs Reward Correlation')
        ax.set_xlabel('Loss')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # 9. 학습 진행 상황 요약
    # 전체 학습 진행 상황을 한눈에 볼 수 있는 요약 정보
    ax = axes[8]
    ax.axis('off')
    summary_text = f"""Training Progress Summary
    
Total Steps: {len(steps)}
Final Loss: {metrics['loss'][-1]:.4f}
Final KL: {metrics['kl'][-1]:.4f}
Final Entropy: {metrics['entropy'][-1]:.4f}

Reward Progress:
"""
    for s, r in zip(reward_steps, reward_values):
        summary_text += f"\n  Step {s}: {r:.4f}"
    
    if reward_values:
        reward_improvement = reward_values[-1] - reward_values[0]
        summary_text += f"\n\nReward Improvement: {reward_improvement:.4f}"
        summary_text += f"\nAvg Reward: {np.mean(reward_values):.4f}"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'trajectory_plot_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 그래프 저장됨: {save_path}")
    
    # 보여주기
    plt.show()

def start_tensorboard(log_dir):
    """텐서보드 시작"""
    print("🚀 텐서보드 시작 중...")
    cmd = f"tensorboard --logdir={log_dir}"
    subprocess.Popen(cmd, shell=True)
    print(f"✅ 텐서보드가 실행되었습니다: http://localhost:6006")

def main():
    # 기본 경로 또는 인자로 받은 경로
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = traj_path
    
    # 경로 확인
    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return
    
    print(f"📂 Trajectory 파일 로드 중: {json_path}")
    
    # 데이터 로드
    data = load_trajectory(json_path)
    print(f"✅ {len(data)}개의 스텝 데이터 로드 완료")
    
    # 저장 디렉토리
    save_dir = os.path.dirname(json_path)
    
    # 그래프 그리기
    plot_trajectories(data, save_dir)
    
    # 텐서보드 시작 (로그 디렉토리가 있다면)
    log_dir = "./logs"
    if os.path.exists(log_dir):
        start_tensorboard(log_dir)
    else:
        print(f"⚠️  텐서보드 로그 디렉토리를 찾을 수 없습니다: {log_dir}")
    
    print("\n💡 팁:")
    print("- 그래프 창을 닫으면 프로그램이 종료됩니다")
    print("- 텐서보드는 백그라운드에서 계속 실행됩니다")
    print("- 텐서보드 종료: Ctrl+C 또는 터미널에서 'pkill tensorboard'")

if __name__ == "__main__":
    main()