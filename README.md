# AttABseq: 단백질 서열 기반 항원-항체 결합 친화도 변화 예측을 위한 어텐션 기반 딥러닝 모델

## 소개

AttABseq는 항체 돌연변이와 관련된 항원-항체 결합 친화도 변화를 예측하기 위한 엔드-투-엔드 서열 기반 딥러닝 모델입니다.

## 사용법

### 1. 환경 설정

먼저, `setup_environment.sh` 스크립트를 실행하여 `attabseq`라는 이름의 conda 환경을 설정하고 필요한 모든 종속성을 설치합니다.

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

설치가 완료되면 다음 명령어를 사용하여 환경을 활성화합니다.

```bash
conda activate attabseq
```

### 2. 모델 학습

`train_model.py` 스크립트를 사용하여 모델을 학습시킬 수 있습니다. 이 스크립트는 데이터셋 CSV 파일의 경로와 교차 검증 분할 수를 인자로 받습니다.

```bash
cd cross_validation/script
python train_model.py --dataset_path <path_to_csv> --n_splits <num_splits>
```

예를 들어, `AB645.csv` 데이터셋으로 10-폴드 교차 검증을 사용하여 모델을 학습시키려면 다음 명령어를 실행합니다.

```bash
python train_model.py --dataset_path ../data/AB645.csv --n_splits 10
```

`--n_splits` 인자는 선택 사항이며, 기본값은 10입니다.

학습된 모델과 결과는 `cross_validation/output<dataset_name>` 디렉토리에 저장됩니다. 예를 들어, `AB645.csv` 데이터셋의 경우 결과는 `cross_validation/outputAB645`에 저장됩니다.

### 3. 예측

학습된 모델을 사용하여 예측을 수행하려면 `predict.py` 스크립트를 사용합니다. 이 스크립트는 학습된 모델의 경로, 예측할 데이터셋의 경로, 그리고 예측 결과를 저장할 경로를 인자로 받습니다.

```bash
python predict.py --model_path <path_to_model> --dataset_path <path_to_csv> --output_path <path_to_output_csv>
```

예를 들어, `AB645` 데이터셋으로 학습된 모델을 사용하여 예측을 수행하고 결과를 `predictions.csv`에 저장하려면 다음 명령어를 실행합니다.

```bash
python predict.py --model_path ../outputAB645/best_pcc_model/model_1 --dataset_path ../data/AB645.csv --output_path ./predictions.csv
```

## 데이터 구조

### 입력 데이터

입력 데이터는 `cross_validation/data` 디렉토리에 있는 CSV 파일입니다. 스크립트는 `antibody_light_seq` 열의 존재 여부로 두 가지 다른 형식을 처리할 수 있습니다.

- **포맷 1 (예: AB645.csv, AB1101.csv):**
  - `PDB`: 단백질 데이터 뱅크 ID.
  - `Mutation`: 돌연변이 정보.
  - `antibody_light_seq`: 항체 경쇄 서열.
  - `antibody_heavy_seq`: 항체 중쇄 서열.
  - `antigen_a_seq`: 항원 A 서열.
  - `antigen_b_seq`: 항원 B 서열.
  - `antibody_light_seq_mut`: 돌연변이된 항체 경쇄 서열.
  - `antibody_heavy_seq_mut`: 돌연변이된 항체 중쇄 서열.
  - `antigen_a_seq_mut`: 돌연변이된 항원 A 서열.
  - `antigen_b_seq_mut`: 돌연변이된 항원 B 서열.
  - `ddG`: 결합 친화도 변화 값 (타겟 변수).

- **포맷 2 (예: S1131.csv):**
  - `PDB`: 단백질 데이터 뱅크 ID.
  - `mutation`: 돌연변이 정보.
  - `a`: 항체 서열.
  - `b`: 항원 서열.
  - `a_mut`: 돌연변이된 항체 서열.
  - `b_mut`: 돌연변이된 항원 서열.
  - `ddG`: 결합 친화도 변화 값 (타겟 변수).

### 출력 데이터

학습 과정의 출력은 `cross_validation/output<dataset_name>` 디렉토리에 저장됩니다. 이 디렉토리에는 다음과 같은 하위 디렉토리가 포함됩니다.

- `loss_min_result`: 검증 손실이 가장 낮은 모델의 성능 지표.
- `loss_min_model`: 검증 손실이 가장 낮은 모델의 가중치.
- `best_pcc_result`: 피어슨 상관 계수가 가장 높은 모델의 성능 지표.
- `best_pcc_model`: 피어슨 상관 계수가 가장 높은 모델의 가중치.
- `best_r2_result`: R-제곱 점수가 가장 높은 모델의 성능 지표.
- `best_r2_model`: R-제곱 점수가 가장 높은 모델의 가중치.

각 `result` 디렉토리의 `RECORD_*.txt` 파일에는 에포크, 시간, 학습 손실, 검증 손실, 피어슨 상관 계수, MAE, MSE, RMSE, R-제곱 등의 정보가 포함됩니다.

## 모델 아키텍처

AttABseq는 인코더-디코더 구조를 기반으로 합니다.

- **인코더**: 1D 컨볼루션 신경망(CNN)을 사용하여 단백질 서열의 특징을 추출합니다. 여러 커널 크기를 가진 여러 CNN 레이어를 사용하여 다양한 길이의 패턴을 포착합니다.
- **디코더**: 어텐션 메커니즘을 사용하여 인코더에서 추출된 항원과 항체 특징 간의 상호작용을 학습합니다. 셀프-어텐션과 인터-어텐션 레이어를 모두 사용하여 서열 내 및 서열 간의 관계를 모델링합니다.
- **예측**: 디코더의 출력을 완전 연결(fully-connected) 레이어에 통과시켜 최종적으로 결합 친화도 변화(ddG) 값을 예측합니다.

## 학습 데이터

이 모델은 제공된 모든 CSV 데이터셋으로 학습할 수 있습니다. 기본적으로 다음 세 가지 데이터셋이 제공됩니다.

- **AB645**: 645개의 항체-항원 복합체에 대한 데이터셋.
- **AB1101**: 1101개의 항체-항원 복합체에 대한 데이터셋.
- **S1131**: 1131개의 항체-항원 복합체에 대한 데이터셋.

각 데이터셋은 교차 검증(cross-validation)을 위해 사용되며, 모델의 일반화 성능을 평가하는 데 도움이 됩니다.