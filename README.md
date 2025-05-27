# SAC 강화학습 기반 주식 트레이딩 시스템

SAC(Soft Actor-Critic) 알고리즘을 활용한 주식 트레이딩 시스템입니다. 이 프로젝트는 PyTorch와 CUDA를 활용하여 GPU 환경에서 강화학습 기반 트레이딩 전략을 개발하고 평가합니다.

## 프로젝트 개요

이 프로젝트는 다음과 같은 주요 기능을 제공합니다:

1. Alpha Vantage API를 통한 주식 데이터 수집
2. 기술적 지표 계산 및 데이터 전처리
3. 단일 및 다중 자산 트레이딩 환경 구현
4. SAC 강화학습 알고리즘 구현 (표준 및 CNN 기반 네트워크)
5. 모델 학습 및 평가 도구

## 설치 방법

### 환경 설정

다음 명령어를 실행하여 필요한 환경을 설정합니다:

```bash
# Windows
setup_env.bat

# Linux/Mac
# bash setup_env.sh
```

또는 수동으로 다음 패키지들을 설치할 수 있습니다:

```bash
# Conda 환경 생성
conda create -n sac_trading_py310 python=3.10 -y
conda activate sac_trading_py310

# 필수 패키지 설치
conda install numpy=1.23.5 -y
conda install pytorch=2.1.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install pandas matplotlib seaborn scikit-learn tqdm requests alpha_vantage pytest pytest-cov python-dotenv
```

### Alpha Vantage API 키 설정

`src/config/config.py` 파일에서 Alpha Vantage API 키를 설정하거나, 환경 변수로 설정할 수 있습니다.

## 사용 방법

### 데이터 수집

```bash
# 기본 설정으로 데이터 수집
python -m src.data_collection.data_collector

# 특정 심볼 데이터 수집
python -m src.data_collection.data_collector --symbols AAPL MSFT GOOGL
```

### 모델 학습

```bash
# 기본 설정으로 단일 자산 학습
python -m src.training.run_training --symbols AAPL --num_episodes 100

# CNN 모델 사용
python -m src.training.run_training --symbols AAPL --use_cnn --num_episodes 100

# 다중 자산 학습
python -m src.training.run_training --symbols AAPL MSFT GOOGL --multi_asset --num_episodes 100
```

### 모델 평가

```bash
# 학습된 모델 평가
python -m src.evaluation.run_evaluation --model_path models/final_sac_model_20230101_120000 --symbols AAPL

# 렌더링 활성화
python -m src.evaluation.run_evaluation --model_path models/final_sac_model_20230101_120000 --symbols AAPL --render
```

## 프로젝트 구조

```
v02_sac-trading/
├── data/                      # 수집된 주식 데이터 저장
├── logs/                      # 로그 파일 저장
├── models/                    # 학습된 모델 저장
├── results/                   # 백테스트 결과 저장
├── docs/                      # 프로젝트 문서
│   ├── diagrams/              # 클래스 다이어그램 등 UML 문서
│   ├── flowcharts/            # 프로세스 플로우차트
│   └── web/                   # 웹 대시보드 관련 문서
├── tests/                     # 단위 테스트 및 통합 테스트
│   └── unit/                  # 단위 테스트
└── src/                       # 소스 코드
    ├── __pycache__/           # 파이썬 캐시 파일
    ├── backtesting/           # 백테스팅 모듈
    ├── config/                # 설정 파일
    ├── dashboard/             # 웹 대시보드 구현
    ├── data_collection/       # 데이터 수집 모듈
    ├── environment/           # 트레이딩 환경 모듈
    ├── evaluation/            # 모델 평가 모듈
    ├── models/                # SAC 모델 구현
    ├── preprocessing/         # 데이터 전처리 모듈
    ├── trading/               # 실시간 트레이딩 모듈
    ├── training/              # 모델 학습 모듈
    └── utils/                 # 유틸리티 함수 및 데이터베이스 관리
```

## 주요 모듈 설명

### 1. 데이터 수집 (`src/data_collection`)
- Alpha Vantage API를 통한 주식 데이터 수집
- 수집된 데이터 저장 및 로드 기능

### 2. 데이터 전처리 (`src/preprocessing`)
- 기술적 지표 계산 (이동평균, MACD, RSI 등)
- 데이터 정규화 및 윈도우 샘플 생성
- 학습/검증/테스트 데이터 분할

### 3. 트레이딩 환경 (`src/environment`)
- 단일 자산 트레이딩 환경 (`TradingEnvironment`)
- 다중 자산 트레이딩 환경 (`MultiAssetTradingEnvironment`)
- 보상 함수 및 포트폴리오 가치 계산

### 4. SAC 모델 (`src/models`)
- Actor 및 Critic 네트워크 구현
- 일반 및 CNN 기반 네트워크 구조
- 경험 리플레이 버퍼 및 모델 저장/로드 기능

### 5. 학습 모듈 (`src/training`)
- SAC 알고리즘 학습 프로세스
- 학습 통계 기록 및 시각화
- 모델 체크포인트 저장

### 6. 평가 모듈 (`src/evaluation`)
- 학습된 모델 성능 평가
- 포트폴리오 가치, 샤프 비율, 최대 낙폭 등 지표 계산
- 평가 결과 시각화 및 저장

## 테스트

단위 테스트를 실행하려면:

```bash
# 모든 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/unit/test_data_collector.py
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 