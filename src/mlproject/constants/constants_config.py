from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
SCHEMA_FILE_PATH = ROOT_DIR / "schema.yaml"
max = ROOT_DIR

print(CONFIG_FILE_PATH)
print(CONFIG_FILE_PATH.exists())