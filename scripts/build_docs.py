import argparse
import subprocess
import sys
from pathlib import Path

from nbclient import NotebookClient
from nbformat import read, write


def run(cmd, **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Fast dev build: do not execute notebooks",
    )

    parser.add_argument(
        "--no-nbconvert",
        action="store_true",
        help="Skip nbconvert step (assumes markdown already exists)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tutorials_dir = project_root / "docs/tutorials"
    static_dir = tutorials_dir / "_static"
    static_dir.mkdir(exist_ok=True)

    notebooks = list(tutorials_dir.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return

    for nb_path in notebooks:
        md_path = static_dir / f"{nb_path.stem}.md"

        if not args.dev:
            print(f"Executing {nb_path}...")
            with nb_path.open("r", encoding="utf-8") as f:
                nb = read(f, as_version=4)

            client = NotebookClient(
                nb, timeout=600, kernel_name=f"python{sys.version_info.major}"
            )
            client.execute(cwd=nb_path.parent)

            with nb_path.open("w", encoding="utf-8") as f:
                write(nb, f)
        else:
            print(f"[DEV] Skipping execution of {nb_path}")

        if not args.no_nbconvert:
            if args.dev and md_path.exists():
                print(f"[DEV] Using existing markdown for {nb_path}")
            else:
                print(f"Exporting {nb_path} to markdown...")
                run(
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
                    ]
                )
        else:
            print(f"[DEV] Skipping nbconvert for {nb_path}")

        if not args.dev:
            print(f"Stripping outputs from {nb_path}...")
            run([sys.executable, "-m", "nbstripout", str(nb_path)])

    print("Building Sphinx docs...")
    run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(project_root / "docs"),
            str(project_root / "docs/_build/html"),
        ]
    )

    print("Done!")
