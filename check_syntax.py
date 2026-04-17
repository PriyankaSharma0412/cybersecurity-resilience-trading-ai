"""Check syntax of all generated scripts."""
import py_compile
import glob
import os

scripts = sorted(glob.glob('scripts/*.py'))
errors = []

for path in scripts:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  OK: {os.path.basename(path)}")
    except py_compile.PyCompileError as e:
        errors.append((path, str(e)))
        print(f"  ERROR: {os.path.basename(path)}: {e}")

print(f"\n{'='*50}")
print(f"Total: {len(scripts)} scripts checked")
print(f"Errors: {len(errors)}")
if errors:
    print("\nFailed scripts:")
    for p, e in errors:
        print(f"  {p}: {e}")
