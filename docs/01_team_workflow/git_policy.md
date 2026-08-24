# Git · GitHub 운영 규칙

개정일 2026-08-24 (1차)
적용 범위 3인 전원
관련 문서 `work_discipline.md` (작업 규율), `../03_decision_log.md`

이 문서는 "어떻게 협업하는가"를 정한다. "무엇을 믿는가"는 `work_discipline.md`가 정한다.

---

## 1. 원칙 세 줄

1. `main`은 항상 동작하는 상태를 유지한다. 직접 push 금지
2. 노트북은 **1인 1파일**. 공동 편집하지 않는다
3. 커밋 단위는 "설명 가능한 하나의 변경"이다 (WD-12)

---

## 2. 브랜치

| 브랜치 | 용도 |
|---|---|
| `main` | 항상 동작하는 기준선. PR 병합으로만 갱신 |
| `feat/<주제>` | 기능 추가 |
| `fix/<주제>` | 버그 수정 |
| `exp/<EXP-NNN>` | 실험. 실험 디렉토리 번호와 일치시킨다 |
| `docs/<주제>` | 문서 |
| `refactor/<주제>` | 동작 변경 없는 정리 |

브랜치명은 소문자 영문 snake_case 또는 하이픈. 한글 금지.
예 — `feat/preprocess_missing`, `exp/EXP-003`, `docs/work_discipline`

수명이 긴 브랜치를 만들지 않는다. 3일 이상 살아 있는 브랜치는 병합하거나 버린다. 실행기에는 하루 단위로 닫는다.

## 3. 커밋

형식은 `타입: 한글 설명`.

타입은 브랜치 타입과 동일한 5종 + `chore`. 설명은 무엇을 왜 바꿨는지 한 문장.

예 — `feat: 결측 구간 선형보간 전처리 추가`
예 — `fix: 시간 분할 시 검증셋 누수 수정`
예 — `docs: 작업 규율 WD-05 그림 경로 정정`

지켜야 할 것.

- **한글 커밋 메시지는 UTF-8 파일로 분리 후 `git commit -F`** (WD-11). 인라인 `-m`은 Windows PowerShell에서 깨질 수 있다
- 커밋에 넣기 전 `git diff --staged`로 실제로 무엇이 들어가는지 본다
- 자동 생성물과 손으로 쓴 변경을 한 커밋에 섞지 않는다
- 번호 재배치(EXP-NNN, KI-NNN)는 단독 커밋 (WD-09)

## 4. 노트북 충돌 대책 — 1인 1파일

`.ipynb`는 JSON이라 두 사람이 같은 파일을 건드리면 셀 출력과 실행 카운트 때문에 반드시 충돌한다. 실행기 14일에 이걸로 시간을 쓰면 손해다.

**채택 방식 — 소유권 분리.**

- 노트북은 `experiments/EXP-NNN_<주제>/` 안에 둔다
- **각 EXP 디렉토리에는 소유자가 한 명이다.** 소유자만 그 안의 노트북을 편집한다
- 다른 사람의 노트북을 고쳐야 하면, 고치지 말고 소유자에게 말한다
- 재사용할 로직은 노트북에 남기지 말고 `src/`로 올린다. `src/`는 .py라 정상적으로 병합된다
- 제출용 노트북은 `submission/`에 단독으로 두고 소유자는 팀장이다

**강제 장치.** `.gitattributes`에 `*.ipynb merge=binary`를 설정했다. 규칙을 어기고 같은 노트북을 양쪽에서 고치면 git이 자동 병합을 거부하고 충돌로 멈춘다. 조용히 깨진 JSON이 커밋되는 것보다 낫다. 충돌이 나면 한쪽을 통째로 고르고, 잃은 작업은 손으로 옮긴다.

**nbstripout은 도입하지 않는다.** 3인이 각자 세팅해야 하는데 한 명만 빠뜨려도 상태가 섞여서 오히려 문제가 커진다. 대신 위의 소유권 규칙으로 해결한다.

## 5. Pull Request

혼자 만든 브랜치도 PR로 병합한다. 리뷰가 목적이 아니라 **기록이 목적**이다. 발표평가에서 "어떤 과정을 거쳤는가"의 근거가 된다.

PR 본문에 아래 네 줄을 넣는다.

> 무엇을 — 한 문장
> 왜 — 결정 근거 또는 관련 EXP/KI 번호
> 검증 — 어떻게 확인했는가 (WD-03)
> 수치 영향 — `results/metrics.json`이 바뀌는가 (예/아니오)

수치가 바뀌는 PR은 문서 갱신을 같이 하거나, 갱신 필요를 PR 본문에 명시한다 (WD-01).

리뷰는 팀장이 본다. 실행기에는 self-merge를 허용하되 PR은 반드시 남긴다.

## 6. 원격 저장소 · 브랜치 보호 (팀장 작업)

**리포지토리는 Private으로 만든다.** 대회 종료 전까지 공개하지 않는다. 부정제출 시비를 만들 이유가 없다.

GitHub에서 빈 리포지토리를 생성한 뒤(README·gitignore 체크 해제) 로컬에서 연결한다.

```
git remote add origin https://github.com/<계정>/kamp_2026.git
git push -u origin main
```

이어서 GitHub 웹에서 보호 규칙을 건다. Settings → Branches → Add branch ruleset (또는 Branch protection rules).

- 대상 브랜치 `main`
- Require a pull request before merging — 켠다
- Require approvals — **0으로 둔다.** 3인 팀에 강제 승인을 걸면 실행기에 서로가 병목이 된다
- Allow force pushes / Allow deletions — 끈다
- Do not allow bypassing the above settings — 켠다. 팀장 본인도 막아야 의미가 있다

팀원 초대는 Settings → Collaborators에서 Write 권한으로 2명.

## 7. 추적 · 비추적 경계

| 경로 | 상태 | 이유 |
|---|---|---|
| `data/` | 비추적 | 원본 데이터셋. 재배포 조건 불명확 + 용량 |
| `outputs/` | 비추적 | 실험 부산물. 언제 지워도 되는 것 |
| `results/` | **추적** | 보고용 확정 산출물. `metrics.json`, `figures/` |
| `submission/` | **추적** | 최종 제출물 |
| `experiments/` | **추적** | 노트북 포함. 과정 기록이 발표 근거다 |
| `docs/reference/` | **추적** | 공고문·과거 회차 원문 |

`results/`를 `.gitignore`에 추가하지 않는다. 여기가 비추적이 되는 순간 WD-01이 무너진다.

데이터 공유는 git이 아니라 별도 경로로 한다. 경로 규약은 `config.py`에 두고, 각자 로컬에 같은 상대 구조로 놓는다.

## 8. 하지 않는 것

세팅 비용이 회수되지 않는 항목이다. 필요해지면 그때 다시 판단한다.

- GitHub Actions CI — 3인 14일 규모에 과하다. `check_report.py`는 로컬에서 수동 실행
- CODEOWNERS — 3인이면 구두로 충분하다
- Git LFS — 현재 최대 파일이 640KB다. hwp/pdf 원문 수준에서는 불필요
- 릴리스 · 태그 — 제출 시점에 한 번만 태그를 남긴다 (`submission-2026-10-08`)

## 9. 일일 루틴 (실행기)

작업 시작 — `git switch main` → `git pull` → 브랜치 생성
작업 종료 — 커밋 → push → PR 생성 → 병합 → 스탠드업에 한 줄

브랜치를 열어둔 채 하루를 넘기지 않는다.
