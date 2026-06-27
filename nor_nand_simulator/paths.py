from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOR_CONFIG  = _ROOT / "nor_gate_params.toml"
DEFAULT_NAND_CONFIG = _ROOT / "nand_gate_params.toml"