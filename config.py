"""저장소 공통 경로와 시드. 시드와 경로는 이 파일에서만 정의한다(AGENTS.md §4)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CONFIGS = PROJECT_ROOT / "configs"
RESULTS = PROJECT_ROOT / "results"
OUTPUTS = PROJECT_ROOT / "outputs"

SEEDS = tuple(range(10))
