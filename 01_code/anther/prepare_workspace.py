"""Copy anther inputs and saved results to a separate working directory."""

import argparse
import json
import shutil
from pathlib import Path

WORKSPACE_DIRECTORIES = (
    "inputs",
    "tables",
    "functional",
    "validation",
    "versions",
    "sources",
)
COPIED_DIRECTORIES = ("inputs", "tables", "functional", "sources")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def resolve_sources(package):
    source_file = package / "04_documentation/anther_source_paths.package_relative.json"
    sources = json.loads(source_file.read_text(encoding="utf-8"))
    return {name: str(package / relative) for name, relative in sources.items()}


def validate_output(path, package):
    if path is None:
        raise ValueError("--output is required.")
    output = path.resolve()
    if output == package or output.is_relative_to(package):
        raise ValueError("Output must be outside the archive.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory.")
    return output


def copy_workspace(package, output, resolved_sources):
    output.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_DIRECTORIES:
        (output / name).mkdir(exist_ok=True)
    for name in COPIED_DIRECTORIES:
        for source in (package / "02_data/anther" / name).iterdir():
            if source.is_file():
                shutil.copy2(source, output / name / source.name)
    (output / "inputs/source_paths.json").write_text(
        json.dumps(resolved_sources, indent=2), encoding="utf-8"
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2]
    resolved_sources = resolve_sources(package)
    missing = [path for path in resolved_sources.values() if not Path(path).is_file()]
    if missing:
        parser.error("Missing packaged inputs: " + ", ".join(missing))
    if args.check:
        print(
            json.dumps({"source_paths_checked": len(resolved_sources), "analysis_started": False})
        )
        return
    try:
        output = validate_output(args.output, package)
    except ValueError as error:
        parser.error(str(error))
    copy_workspace(package, output, resolved_sources)
    print(json.dumps({"workspace": str(output), "analysis_started": False}))


if __name__ == "__main__":
    main()
