"""Package the server into a standalone executable (no Python needed on target).

Usage:
  python build.py          # 1. Build the index first
  python package.py        # 2. Package into dist/ folder
  # Then copy the dist/ folder to any machine and run rag-server.exe
"""

import shutil
import subprocess
import sys
from pathlib import Path

DIST_DIR = Path("dist")
EXE_NAME = "rag-server"


def main():
    # 1. Check prerequisites
    if not Path("index.pkl").exists():
        print("Error: index.pkl not found. Run 'python build.py' first.")
        sys.exit(1)

    print("Building standalone executable with PyInstaller...")
    print(f"  Source: server.py")
    print(f"  Output: {DIST_DIR}/{EXE_NAME}.exe (or {EXE_NAME} on Linux)")

    # 2. Run PyInstaller
    # --onefile = single .exe file
    # --hidden-import = uvicorn internals that PyInstaller might miss
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", EXE_NAME,
        "--clean",
        "--noconfirm",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.lifespan.on",
        "server.py",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nPyInstaller failed. Make sure it's installed: pip install pyinstaller")
        sys.exit(1)

    # 3. Copy index.pkl into dist/ so the exe can find it
    shutil.copy2("index.pkl", DIST_DIR / "index.pkl")
    print(f"  Copied index.pkl → {DIST_DIR}/index.pkl")

    # 4. Done
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Build complete!

  Deliverables (in {DIST_DIR}/):
    {EXE_NAME}.exe   ← double-click to start
    index.pkl        ← the knowledge index

  To deploy on another machine:
    1. Copy the whole {DIST_DIR}/ folder
    2. Create .env inside it:
         DEEPSEEK_API_KEY=sk-xxx
    3. Double-click {EXE_NAME}.exe
    4. Server starts at http://0.0.0.0:8000

  Test: curl http://localhost:8000/ask -d '{{"question":"hello"}}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
