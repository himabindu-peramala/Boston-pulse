import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd()))
from src.validation.schema_registry import SchemaRegistry
from src.shared.config import get_config


def register_all_schemas():
    config = get_config("dev")
    registry = SchemaRegistry(config)

    # Absolute path inside the Docker container
    schemas_dir = Path("/opt/airflow/data-pipeline/schemas")

    if not schemas_dir.exists():
        # Try relative to current file if absolute fails
        schemas_dir = Path(__file__).parent / "schemas"

    print(f"Syncing schemas from: {schemas_dir}")

    for dataset_dir in schemas_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        for schema_file in dataset_dir.glob("*.json"):
            filename = schema_file.name
            if "features" in filename:
                layer = "features"
            elif "processed" in filename:
                layer = "processed"
            elif "raw" in filename:
                layer = "raw"
            else:
                continue

            print(f"Registering {dataset}/{layer} from {schema_file}...")

            with open(schema_file, "r") as f:
                schema_content = json.load(f)

            # Extracts only the "properties" part as "schema" for register_schema
            # because register_schema expects the dict of columns
            props = schema_content.get("properties", {})

            registry.register_schema(
                dataset=dataset,
                layer=layer,
                schema=props,
                version="v1_fix",
                description=f"Fix for {dataset} {layer} schema",
            )
            print(f"Successfully registered {dataset}/{layer}")


if __name__ == "__main__":
    register_all_schemas()
