## 빠른 시작

### 옵션 1: VS Code 작업 (권장)
`Ctrl+Shift+P`를 누르고 "Tasks: Run Task"를 입력한 후 다음 중 선택:
- **AttABseq: Show Training Summary** - 이 가이드 보기
- **AttABseq: Interactive Training Launcher** - 대화형 훈련 시작
- **AttABseq: Setup Training Environment** - 설정만 실행
- **AttABseq: Train All Datasets** - 모든 데이터셋 훈련
- **AttABseq: Train AB1101 Dataset** - 특정 데이터셋 훈련
- **AttABseq: Train AB645 Dataset** - 특정 데이터셋 훈련
- **AttABseq: Train S1131 Dataset** - 특정 데이터셋 훈련

### 옵션 2: 명령줄
```bash
# 대화형 모드 (초보자 권장)
python start_training.py

# 명령줄 모드
python setup_training.py --dataset all       # 모든 데이터셋 훈련
python setup_training.py --dataset AB1101    # 특정 데이터셋 훈련
python setup_training.py --setup-only        # 설정만 실행
```

## 포함된 내용

✅ **Python 환경**: 모든 종속성이 포함된 가상 환경  
✅ **훈련 스크립트**: Windows용으로 업데이트 및 경로 수정  
✅ **출력 디렉토리**: 모든 데이터셋에 대해 자동 생성  
✅ **모델 파일**: 적절히 연결되고 이름 지정됨  
✅ **데이터 파일**: 검증되고 접근 가능  
✅ **VS Code 통합**: 쉬운 실행을 위한 작업  

## 사용 가능한 데이터셋

- **AB1101**: 1,101개의 항체-항원 결합 돌연변이
- **AB645**: 645개의 항체-항원 결합 돌연변이  
- **S1131**: 1,131개의 단백질-단백질 상호작용 돌연변이

## 훈련 기능

- 🧠 어텐션 기반 딥러닝 모델
- 📊 5-폴드 교차 검증
- 📈 다중 평가 지표 (PCC, R², MAE, RMSE)
- ⏹️ 과적합 방지를 위한 조기 종료
- 💾 자동 모델 및 결과 저장

## 예상 출력

훈련은 `cross_validation/output*/`에 정리된 결과를 생성합니다:
- `RECORD_*.txt`: 각 에포크의 훈련 지표
- `model_*`: 저장된 모델 가중치
- 세 가지 모델 변형: `loss_min`, `best_pcc`, `best_r2`

## 하드웨어 요구사항

- **최소**: 8GB RAM, 10GB 디스크 공간
- **권장**: CUDA를 지원하는 GPU, 16GB RAM
- **훈련 시간**: 데이터셋당 2-6시간 (GPU) 또는 10-30시간 (CPU)

## 도움이 필요하신가요?

- 📖 자세한 문서는 `TRAINING_GUIDE.md`를 읽어보세요
- 🚀 빠른 참조를 위해 `python training_summary.py`를 실행하세요
- 🔧 모든 종속성은 자동으로 관리됩니다
