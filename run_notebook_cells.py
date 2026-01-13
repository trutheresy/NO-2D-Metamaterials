"""
Script to execute notebook cells and report errors.
This will help identify and fix issues in the notebook.
"""

import json
import sys
from pathlib import Path

# Load notebook
notebook_path = Path('compare_matlab_files.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells")
print("=" * 80)

# Execute cells that contain code
exec_globals = {}
exec_locals = {}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if source.strip():  # Only execute non-empty cells
            print(f"\n{'='*80}")
            print(f"Executing Cell {i}")
            print(f"{'='*80}")
            print(f"Source preview (first 200 chars):\n{source[:200]}...")
            print(f"\nExecuting...")
            try:
                exec(source, exec_globals, exec_locals)
                print(f"✓ Cell {i} executed successfully")
            except Exception as e:
                print(f"✗ ERROR in Cell {i}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                print(f"\nStopping execution at Cell {i}")
                break

print(f"\n{'='*80}")
print("Notebook execution complete")
print(f"{'='*80}")

