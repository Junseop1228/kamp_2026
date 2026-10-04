# KI-002. MCP 경유 conda 실행에서 프로세스와 오류 출력이 사라짐

- **태그** #env
- **등록일** 2026-10-04
- **상태** 닫힘 (우회 확정)
- **출처** P0 실행 환경 구성
- **관련 문서** `../docs/03_decision_log.md` D-005

## 무엇이 일어났나

1. 백그라운드 실행(`Start-Process conda ... -NoNewWindow`)으로 환경 생성을 시작했다. MCP 호출이 타임아웃되면서 conda가 같이 종료됐다(로그 끝 "Terminate batch job"). 환경은 만들어지지 않았다.
2. `conda run`으로 검증 스크립트를 돌렸는데 MCP 응답에 아무것도 나오지 않았다. 실제로는 import 오류로 실패한 상태였다. 출력이 없어서 정상처럼 보였다.

## 원인 (추정)

1. Windows의 `conda`는 배치 파일 래퍼다. 같은 콘솔에 붙여 띄우면 MCP가 호출을 끊을 때 함께 중단된다.
2. `conda run`의 stderr가 MCP 응답으로 전달되지 않는다.
3. 스크립트를 `outputs/`에서 실행해 루트 `config.py`를 import하지 못했다.

## 조치

장시간 작업은 `conda.exe`를 직접, 숨김 창으로 띄우고 로그로 완료를 확인한다. MCP가 타임아웃을 반환해도 프로세스는 끝까지 돈다. `env create`로 확인했고, `run` 조합은 아직 실측 전이다.

```powershell
Start-Process -FilePath "$env:USERPROFILE\anaconda3\Scripts\conda.exe" -ArgumentList "run","-n","kamp2026","--no-capture-output","python","run.py" -WorkingDirectory $PWD -RedirectStandardOutput outputs\logs\run.out.log -RedirectStandardError outputs\logs\run.err.log -WindowStyle Hidden -PassThru
```

짧은 실행은 stderr까지 받는다. repo 루트 기준 import가 필요하면 `PYTHONPATH`를 지정한다.

```powershell
$env:PYTHONPATH="$PWD"; $out = & conda run -n kamp2026 --no-capture-output python script.py 2>&1; $out
```

## 교훈

출력이 없는 것을 성공으로 읽지 않는다. 검증 명령은 기대 수치를 반드시 출력하게 만들고, 출력이 없으면 실패로 본다.
