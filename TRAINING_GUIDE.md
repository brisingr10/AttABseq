# AttABseq 훈련 설정 가이드

이 가이드는 AttABseq 훈련 파이프라인 설정 및 실행에 대한 단계별 지침을 제공합니다.

## 빠른 시작

### 1. 환경 설정
```bash
# Python 환경이 이미 구성되어 있습니다
# 종속성은 자동으로 설치됩니다
```

### 2. 훈련 시작 (대화형)
```bash
python start_training.py
```

### 3. 훈련 시작 (명령줄)
```bash
# 모든 데이터셋 설정 및 훈련
python setup_training.py --dataset all

# 설정만 실행 (훈련 없음)
python setup_training.py --setup-only

# 특정 데이터셋 훈련
python setup_training.py --dataset AB1101
python setup_training.py --dataset AB645
python setup_training.py --dataset S1131
```

## 프로젝트 구조

```
AttABseq/
├── README.md                     # 원본 프로젝트 README
├── requirements.txt              # Python 종속성
├── start_training.py            # 대화형 훈련 런처
├── setup_training.py            # 훈련 설정 및 실행기
├── cross_validation/            # 메인 훈련 디렉토리
│   ├── data/                    # 훈련 데이터셋
│   │   ├── AB1101.csv
│   │   ├── AB645.csv
│   │   └── S1131.csv
│   ├── script/                  # 훈련 스크립트
│   │   ├── main_AB1101.py       # AB1101 훈련 스크립트
│   │   ├── main_AB645.py        # AB645 훈련 스크립트
│   │   ├── main_S1131.py        # S1131 훈련 스크립트
│   │   ├── model_AB1101.py      # AB1101 모델 정의
│   │   ├── model_AB645.py       # AB645 모델 정의
│   │   ├── model_S1131.py       # S1131 모델 정의
│   │   ├── predict.py           # 예측 스크립트
│   │   └── ...
│   ├── output1101/              # AB1101 훈련 출력
│   ├── output645/               # AB645 훈련 출력
│   ├── output1131/              # S1131 훈련 출력
│   └── ncbi-blast-2.12.0+/      # BLAST 데이터베이스 (사용 가능한 경우)
└── ...
```

## 훈련 과정

### 훈련이 수행하는 작업

1. **데이터 로딩**: CSV 파일에서 단백질 서열 로드
2. **특성 추출**: 
   - 서열을 원-핫 인코딩으로 변환
   - BLAST를 사용하여 PSSM (위치별 점수 매트릭스) 생성
3. **모델 훈련**: 
   - 어텐션 기반 딥러닝 아키텍처 사용
   - 5-폴드 교차 검증
   - 과적합 방지를 위한 조기 종료
4. **모델 평가**: 
   - 피어슨 상관계수
   - R² 점수
   - 평균 절대 오차 (MAE)
   - 평균 제곱근 오차 (RMSE)

### 출력 파일

각 데이터셋과 폴드에 대해 훈련은 다음을 생성합니다:

- `RECORD_{fold}.txt`: 각 에포크의 훈련 지표
- `model_{fold}`: 저장된 모델 가중치
- 세 가지 유형의 저장된 모델:
  - `loss_min_*`: 최고 검증 손실
  - `best_pcc_*`: 최고 피어슨 상관관계
  - `best_r2_*`: 최고 R² 점수

### 훈련 매개변수

- **배치 크기**: 8
- **학습률**: 0.00001
- **숨겨진 차원**: 256
- **레이어 수**: 3
- **어텐션 헤드**: 8
- **드롭아웃**: 0.1
- **최대 에포크**: 150
- **조기 종료 인내**: 7

## 데이터셋

### AB1101 (항체-항원 결합)
- 1,101개의 돌연변이 기록
- 항체 경쇄/중쇄 및 항원 서열
- ΔΔG 결합 친화도 변화

### AB645 (항체-항원 결합)
- 645개의 돌연변이 기록
- AB1101과 유사한 구조

### S1131 (단백질-단백질 상호작용)
- 1,131개의 돌연변이 기록
- 단백질 A 및 B 서열
- ΔΔG 상호작용 변화

## 요구사항

### 시스템 요구사항
- Python 3.7+
- CUDA 호환 GPU (권장)
- 8GB+ RAM
- 출력용 10GB+ 디스크 공간

### Python 종속성
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.20.0,<2.0.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- seaborn>=0.11.0
- matplotlib>=3.5.0
- networkx>=2.6.0
- xarray>=0.20.0

### 선택적 종속성
- NCBI BLAST+ (PSSM 생성용)

## 문제 해결

### 일반적인 문제

1. **CUDA/GPU 문제**
   - 코드가 자동으로 CUDA 가용성을 감지합니다
   - CUDA를 사용할 수 없으면 CPU로 대체
   - CPU에서 훈련이 더 느려집니다

2. **BLAST 데이터베이스 누락**
   - PSSM 생성에는 BLAST 데이터베이스가 필요합니다
   - BLAST를 사용할 수 없으면 훈련이 실패할 수 있습니다
   - 사용 가능한 경우 미리 계산된 특성 사용을 고려하세요

3. **메모리 문제**
   - 훈련 스크립트에서 배치 크기를 줄이세요
   - 더 작은 모델 차원을 사용하세요
   - 다른 애플리케이션을 닫으세요

4. **경로 문제**
   - 모든 스크립트가 올바른 디렉토리에서 실행되는지 확인하세요
   - `cross_validation/data/`에 데이터 파일이 있는지 확인하세요

### 디버그 모드

훈련을 실행하지 않고 설정을 확인하려면:
```bash
python setup_training.py --setup-only --dry-run
```

## 고급 사용법

### 사용자 정의 훈련 매개변수

다음을 수정하려면 훈련 스크립트를 편집하세요:
- 학습률
- 배치 크기
- 모델 아키텍처
- 훈련 에포크

### 새 데이터셋 추가

1. `cross_validation/data/`에 CSV 파일 추가
2. 해당하는 훈련 스크립트 생성
3. `setup_training.py` 구성 업데이트

### 모델 예측

훈련 후 예측 스크립트 사용:
```bash
cd cross_validation/script
python predict.py
```

## 지원

문제나 질문이 있으시면:
1. 문제 해결 섹션을 확인하세요
2. 모든 종속성이 설치되어 있는지 확인하세요
3. 데이터 파일이 존재하고 접근 가능한지 확인하세요

## 인용

연구에서 AttABseq를 사용하시면 원본 논문을 인용해 주세요:
[여기에 인용 정보 추가]
