import subprocess
import sys
from pathlib import Path

from nbclient import NotebookClient
from nbformat import read, write


def main():
    project_root = Path(__file__).resolve().parent.parent
    tutorials_dir = project_root / "docs/tutorials"
    static_dir = tutorials_dir / "_static"
    static_dir.mkdir(exist_ok=True)

    notebooks = list(tutorials_dir.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return

    for nb_path in notebooks:
        print(f"Executing {nb_path}...")
        with nb_path.open("r", encoding="utf-8") as f:
            nb = read(f, as_version=4)

        client = NotebookClient(
            nb, timeout=600, kernel_name=f"python{sys.version_info.major}"
        )
        client.execute(cwd=nb_path.parent)

        with nb_path.open("w", encoding="utf-8") as f:
            write(nb, f)

        print(f"Exporting {nb_path} to markdown...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                str(nb_path),
                "--output-dir",
                str(static_dir),
            ],
            check=True,
        )

        print(f"Stripping outputs from {nb_path}...")
        subprocess.run([sys.executable, "-m", "nbstripout", str(nb_path)], check=True)

    print("Building Sphinx docs...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(project_root / "docs"),
            str(project_root / "docs/_build/html"),
        ],
        check=True,
    )
    print("Done!")
