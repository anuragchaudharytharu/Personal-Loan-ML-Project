from pathlib import Path

# this file is 3 level down from the root directory.
# So, we go up by 3 level i.e Path(__file__).resolve().parents[3] to get to the project root directory
ROOT_DIR = Path(__file__).resolve().parents[3]

print(ROOT_DIR)

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
SCHEMA_FILE_PATH = ROOT_DIR / "schema.yaml"

print(CONFIG_FILE_PATH)
print(CONFIG_FILE_PATH.exists())