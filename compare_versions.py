"""
Compare old (main) vs new (current branch) behavior.
This script temporarily checks out main to test the old version.
"""
import subprocess
import sys
from pathlib import Path

def run_git_command(cmd):
    """Run a git command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def test_version(version_name):
    """Test the current version."""
    print(f"\n{'='*70}")
    print(f"TESTING {version_name}")
    print(f"{'='*70}")
    
    # Run signature test script
    result = subprocess.run(
        [sys.executable, "test_signature_analysis.py"],
        capture_output=True,
        text=True,
        cwd="C:\\Users\\z665206\\Documents\\PhD\\code\\paradigma"
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    """Main comparison function."""
    print("\n" + "#"*70)
    print("# BACKWARD COMPATIBILITY COMPARISON: OLD vs NEW")
    print("#"*70)
    
    # Test NEW version first
    print("\n[1/3] Testing NEW version (current branch)...")
    new_success = test_version("NEW VERSION (with datetime support)")
    
    # Get current branch
    current_branch, _, _ = run_git_command("git branch --show-current")
    print(f"\nCurrent branch: {current_branch}")
    
    # Stash changes
    print("\n[2/3] Checking out main branch...")
    stdout, stderr, code = run_git_command("git stash")
    if code != 0:
        print(f"Warning: git stash failed: {stderr}")
    
    stdout, stderr, code = run_git_command("git checkout main")
    if code != 0:
        print(f"Error: Could not checkout main: {stderr}")
        return
    
    # Reinstall package to ensure old code is used
    print("Reinstalling package to use main branch code...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd="C:\\Users\\z665206\\Documents\\PhD\\code\\paradigma",
        capture_output=True,
        text=True
    )
    
    # Test OLD version
    print("\n[2/3] Testing OLD version (from main)...")
    old_success = test_version("OLD VERSION (without datetime support)")
    
    # Restore new version
    print("\n[3/3] Restoring new version...")
    stdout, stderr, code = run_git_command(f"git checkout {current_branch}")
    if code != 0:
        print(f"Error: Could not checkout {current_branch}: {stderr}")
    
    stdout, stderr, code = run_git_command("git stash pop")
    if code != 0:
        print(f"Note: git stash pop had status: {code}")
    
    # Reinstall new version
    print("Reinstalling package to use new code...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd="C:\\Users\\z665206\\Documents\\PhD\\code\\paradigma",
        capture_output=True,
        text=True
    )
    
    # Summary
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print(f"Old version (main): {'OK' if old_success else 'FAILED'}")
    print(f"New version (current): {'OK' if new_success else 'FAILED'}")


if __name__ == "__main__":
    main()
