"""Synchronize the documentation copy of the headless workflow notebook."""

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "headless_batch_workflow.ipynb"
DESTINATION = ROOT / "docs" / "examples" / "headless_batch_workflow.ipynb"


def main() -> None:
    """Normalize cell IDs and copy the notebook into the Sphinx source tree."""
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    changed = False
    for index, cell in enumerate(notebook.get("cells", [])):
        if not cell.get("id"):
            source = "".join(cell.get("source", []))
            identity = f"{index}:{cell.get('cell_type', 'cell')}:{source}"
            cell["id"] = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
            changed = True

    if changed:
        SOURCE.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DESTINATION)
    print(f"Synchronized {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
