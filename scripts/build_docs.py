import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nbclient import NotebookClient
from nbformat import read, write


def run(cmd, **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


def clean_style_blocks(content: str) -> str:
    """Remove <style scoped>...</style> blocks from markdown content.

    GitHub markdown doesn't support style tags, so they render as literal text.
    """
    pattern = r"<style scoped>\s*\n.*?\n</style>\n"
    return re.sub(pattern, "", content, flags=re.DOTALL)


def get_python_deps_mtime(project_root: Path) -> float:
    """Get the most recent modification time of all Python source files.

    Returns the mtime (modification time) of the most recently changed .py file
    in src/ and scripts/ directories. This is used to detect if any dependencies
    have changed since the last notebook execution.
    """
    max_mtime = 0
    for pattern in [project_root / "src/**/*.py", project_root / "scripts/**/*.py"]:
        for py_file in project_root.glob(str(pattern.relative_to(project_root))):
            if py_file.is_file():
                max_mtime = max(max_mtime, os.path.getmtime(py_file))
    return max_mtime


def should_execute_notebook(
    nb_path: Path, md_path: Path, deps_mtime: float, force: bool = False
) -> bool:
    """Determine if notebook should be executed.

    Returns True if:
    - force flag is set (--force)
    - markdown doesn't exist yet
    - notebook is newer than markdown
    - any .py dependency is newer than notebook
    """
    if force:
        return True

    if not md_path.exists():
        return True

    nb_mtime = os.path.getmtime(nb_path)
    md_mtime = os.path.getmtime(md_path)

    # Re-execute if notebook is newer than markdown
    if nb_mtime > md_mtime:
        return True

    # Re-execute if any Python dependency is newer than notebook
    if deps_mtime > nb_mtime:
        return True

    return False


def execute_notebook(nb_path: Path, kernel_name: str) -> tuple[Path, bool]:
    """Execute a single notebook. Returns (path, success)."""
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = read(f, as_version=4)

        client = NotebookClient(nb, timeout=600, kernel_name=kernel_name)
        client.execute(cwd=nb_path.parent)

        with nb_path.open("w", encoding="utf-8") as f:
            write(nb, f)

        return nb_path, True
    except Exception as e:
        print(f"Error executing {nb_path}: {e}")
        return nb_path, False


def main():
    parser = argparse.ArgumentParser(
        description="Build documentation with notebook execution and Sphinx"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Fast dev build: do not execute notebooks, use cached markdown",
    )

    parser.add_argument(
        "--no-nbconvert",
        action="store_true",
        help="Skip nbconvert step (assumes markdown already exists)",
    )

    parser.add_argument(
        "--notebook",
        type=str,
        help="Build only a specific notebook (e.g., 'pipeline_orchestrator.ipynb')",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of all notebooks, skip incremental build check",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers for notebook execution (default: 1)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tutorials_dir = project_root / "docs/tutorials"
    static_dir = tutorials_dir / "_static"
    static_dir.mkdir(exist_ok=True)

    if args.notebook:
        # Build only the specified notebook
        nb_path = tutorials_dir / args.notebook
        if not nb_path.exists():
            print(f"Error: Notebook '{args.notebook}' not found in {tutorials_dir}")
            return
        notebooks = [nb_path]
        print(f"Building single notebook: {args.notebook}")
    else:
        # Build all notebooks
        notebooks = list(tutorials_dir.glob("*.ipynb"))
        if not notebooks:
            print("No notebooks found")
            return

    # Get modification time of Python dependencies
    deps_mtime = get_python_deps_mtime(project_root)
    kernel_name = f"python{sys.version_info.major}"
    notebooks_to_execute = []
    notebooks_to_skip = []

    # Determine which notebooks need execution
    if not args.dev:
        for nb_path in notebooks:
            md_path = static_dir / f"{nb_path.stem}.md"
            if should_execute_notebook(nb_path, md_path, deps_mtime, args.force):
                notebooks_to_execute.append(nb_path)
            else:
                notebooks_to_skip.append(nb_path.name)

        if notebooks_to_skip:
            print(
                f"[INCREMENTAL] Skipping {len(notebooks_to_skip)} notebook(s) "
                f"(no changes detected)"
            )
            for nb_name in notebooks_to_skip:
                print(f"  - {nb_name}")

        if notebooks_to_execute:
            print(f"Executing {len(notebooks_to_execute)} notebook(s)...")
            if args.parallel > 1:
                print(f"  Using {args.parallel} parallel workers")

            # Execute notebooks in parallel if requested
            if args.parallel > 1:
                with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                    futures = {
                        executor.submit(execute_notebook, nb, kernel_name): nb
                        for nb in notebooks_to_execute
                    }
                    for future in as_completed(futures):
                        nb_path, success = future.result()
                        status = "✓" if success else "✗"
                        print(f"  {status} {nb_path.name}")
            else:
                # Sequential execution
                for nb_path in notebooks_to_execute:
                    print(f"Executing {nb_path.name}...")
                    nb_path_result, success = execute_notebook(nb_path, kernel_name)
                    if not success:
                        print(f"Failed to execute {nb_path.name}")
    else:
        print("[DEV] Skipping notebook execution")
        notebooks_to_execute = []

    # Convert notebooks to markdown
    if not args.no_nbconvert:
        notebooks_to_convert = notebooks_to_execute
        if args.dev:
            notebooks_to_convert = [
                nb for nb in notebooks if not (static_dir / f"{nb.stem}.md").exists()
            ]

        if notebooks_to_convert:
            print(f"Converting {len(notebooks_to_convert)} notebook(s) to markdown...")
            for nb_path in notebooks_to_convert:
                md_path = static_dir / f"{nb_path.stem}.md"
                print(f"  {nb_path.name}")

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

                # Clean style blocks from generated markdown
                if md_path.exists():
                    with open(md_path, encoding="utf-8") as f:
                        content = f.read()
                    cleaned = clean_style_blocks(content)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
        else:
            print("[INCREMENTAL] No notebooks to convert (markdown is up-to-date)")
    else:
        print("[DEV] Skipping nbconvert")

    print("Building Sphinx documentation...")
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

    # Strip outputs AFTER Sphinx has built the docs
    if not args.dev and notebooks_to_execute:
        print(f"Stripping outputs from {len(notebooks_to_execute)} notebook(s)...")
        for nb_path in notebooks_to_execute:
            run([sys.executable, "-m", "nbstripout", str(nb_path)])

    print("✓ Done!")
