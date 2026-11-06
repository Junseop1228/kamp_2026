# KI-003. 사용자 전역 패키지 폴더가 conda 환경 패키지를 가림

- **태그** #env
- **등록일** 2026-10-04
- **상태** 닫힘 (환경 변수로 차단)
- **출처** P1 단위 2 테스트 실행
- **관련 문서** `../docs/03_decision_log.md` D-005, `../environment.yml`

## 무엇이 일어났나

`python -m pytest`가 테스트를 수집하기도 전에 `ModuleNotFoundError: No module named 'platformdirs.pytest_plugin'`으로 죽었다. 환경에 설치된 platformdirs 4.12.2에는 해당 모듈이 있는데도 실패했다.

## 원인

파이썬은 사용자 전역 패키지 폴더(`%APPDATA%\Python\Python311\site-packages`)를 환경의 site-packages보다 앞에 둔다. 이 폴더의 구버전 platformdirs 4.3.8이 먼저 import됐다. 환경 쪽 platformdirs가 등록한 pytest 플러그인 진입점은 구버전에 없는 모듈을 가리켰다.

## 영향 범위

numpy·pandas·scikit-learn·matplotlib·pytest는 전역 폴더에 없어서 환경 쪽이 로드됐다(`__file__` 경로로 확인). P0~P1 수치는 오염되지 않았다. 다만 같은 파이썬 버전을 쓰는 다른 PC에서는 어떤 패키지가 가려질지 알 수 없다.

## 조치

`environment.yml`에 `variables: PYTHONNOUSERSITE: "1"`을 넣어, 환경을 활성화할 때 전역 폴더를 끈다. 이미 만든 환경에는 `conda env config vars set`으로 같은 값을 적용했다.

## 교훈

환경을 고정해도 환경 밖에서 끼어드는 경로가 있다. 패키지 출처는 버전 번호가 아니라 import 경로로 확인한다.
