# AttABseq: 단백질 서열 기반 항원-항체 결합 친화도 변화를 위한 어텐션 기반 딥러닝 예측 방법

## 소개

AttABseq는 항체 돌연변이와 연관된 항원-항체 결합 친화도 변화를 예측하기 위한 종단 간 서열 기반 딥러닝 모델입니다.

## 파일 아키텍처
```
AttABseq
├── analysis
├── attention_ablation
│   ├── k-cv_no-attention
│   │   ├── 645analysis
│   │   ├── 1101analysis
│   │   ├── 1131analysis
│   │   ├── data
│   │   │   ├──AB645.csv
│   │   │   ├──AB645order.csv
│   │   │   ├──AB1101.csv
│   │   │   ├──AB1101order.csv
│   │   │   ├──S1131.csv
│   │   │   └──S1131order.csv
│   │   ├── ncbi-blast-2.12.0+
│   │   ├── output645
│   │   │   ├──best_pcc_model
│   │   │   ├──best_pcc_result
│   │   │   ├──best_r2_model
│   │   │   ├──best_r2_result
│   │   │   ├──loss_min_model
│   │   │   ├──loss_min_result
│   │   ├── output1101
│   │   ├── output1131
│   │   ├── script
│   │   │   ├──main_AB645.py
│   │   │   ├──main_AB1101.py
│   │   │   ├──main_S1131.py
│   │   │   ├──model_AB645.py
│   │   │   ├──model_AB1101.py
│   │   │   ├──model_S1131.py
│   │   │   ├──predict.py
│   │   │   ├──lookahead.py
│   │   │   ├──Radam.py
│   │   │   └──pytorchtools.py
│   └── split_no-attention
├── cross_validation
│   ├── 645analysis
│   ├── 1101analysis
│   ├── 1131analysis
│   ├── data
│   ├── ncbi-blast-2.12.0+
│   ├── output645
│   ├── output1101
│   ├── output1131
│   └── script
├── split
│   ├── 645analysis
│   ├── 1101analysis
│   ├── 1131analysis
│   ├── data
│   ├── ncbi-blast-2.12.0+
│   ├── output645
│   ├── output1101
│   ├── output1131
│   └── script
├── interpretability
│   ├── scatter.py
│   ├── 645
│   │   ├── AttABseq_split_645.csv
│   │   ├── scatter.py
│   │   └── split-645.png
│   ├── 1101
│   ├── 1131
│   ├── interpretable
│   │   ├── 645
│   │   │   ├── interpre_csv
│   │   │   ├── interpre_heatmap
│   │   │   ├── 645_interpre.csv
│   │   │   ├── ab16.txt
│   │   │   ├── ab_mut16.txt
│   │   │   ├── ag16.txt
│   │   │   ├── ag_mut16.txt
│   │   │   ├── AB645.csv
│   │   │   ├── interpre.csv
│   │   │   ├── interpretable.csv
│   │   │   ├── interpre.py
│   │   │   └── split.txt
│   │   ├── 1101
│   │   └── 1131
```
## 사용법

### 1. 환경
- python 3.7.0+
- pytorch 2.0.0+
- torchvision 0.15.0+
- numpy 1.20.0+
- pandas 1.3.0+
- scikit-learn 1.0.0+
- scipy 1.7.0+
- seaborn 0.11.0+
- matplotlib 3.5.0+
- networkx 2.6.0+
- xarray 0.20.0+

### 2. 데이터
- k-cv: AB645.csv / AB1101.csv / S1131.csv
- 레이블 오름차순 정렬 분할: AB645order.csv / AB1101order.csv / S1131order.csv

### 3. 훈련
```
# 가상 환경 활성화 (이미 구성됨)
python setup_training.py --dataset all
```
결과는 "output" 폴더에서 찾을 수 있습니다.

### 4. 테스트
```
# 가상 환경 활성화 (이미 구성됨)
cd cross_validation/script
python predict.py
```
결과는 "output" 폴더에서 찾을 수 있습니다.

### 5. 해석가능성 분석
```
# 가상 환경 활성화 (이미 구성됨)
cd interpretability/interpretable/645
python interpre.py
```
결과는 "interpre_csv" 및 "interpre_heatmap" 폴더에서 찾을 수 있습니다.

## 빠른 시작 (업데이트됨)

### 대화형 모드 (권장)
```bash
python start_training.py
```

### 명령줄 모드
```bash
# 모든 데이터셋 훈련
python setup_training.py --dataset all

# 특정 데이터셋 훈련
python setup_training.py --dataset AB1101
python setup_training.py --dataset AB645
python setup_training.py --dataset S1131

# 설정만 실행
python setup_training.py --setup-only
```

### VS Code 작업 (권장)
`Ctrl+Shift+P` → "Tasks: Run Task" → AttABseq 작업 선택

자세한 정보는 `TRAINING_GUIDE.md` 및 `SETUP_COMPLETE.md`를 참조하세요.

