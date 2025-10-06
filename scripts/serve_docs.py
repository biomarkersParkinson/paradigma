import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    build_dir = project_root / "docs/_build/html"

    if not build_dir.exists():
        print(f"Error: {build_dir} does not exist. Run build_docs.py first.")
        sys.exit(1)

    # Change working directory to the build folder
    os.chdir(build_dir)
    print(f"Serving {build_dir} at http://0.0.0.0:8000")
    try:
        # Optional: open browser automatically
        webbrowser.open("http://localhost:8000")

        # Python 3.7+
        handler_class = SimpleHTTPRequestHandler
        server = HTTPServer(("0.0.0.0", 8000), handler_class)
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
