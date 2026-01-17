#!/usr/bin/env python3
"""List check, test, and .md files sorted by creation date."""

import os
import glob
from pathlib import Path
from datetime import datetime

# Find all relevant files
check_files = list(Path('.').rglob('*check*.py'))
test_files = list(Path('.').rglob('*test*.py'))
md_files = list(Path('.').rglob('*.md'))

# Combine and filter
all_files = check_files + test_files + md_files
all_files = [f for f in all_files 
             if 'node_modules' not in str(f) 
             and '.git' not in str(f)
             and f.exists()]

# Get creation dates
files_with_dates = []
for f in all_files:
    try:
        ctime = os.path.getctime(f)
        files_with_dates.append((str(f), ctime))
    except:
        pass

# Sort by creation date
files_with_dates.sort(key=lambda x: x[1])

# Print
print("=" * 100)
print("Check, Test, and Documentation Files (Sorted by Creation Date)")
print("=" * 100)
print()

current_date = None
for filepath, ctime in files_with_dates:
    date_str = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
    date_only = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d")
    
    if date_only != current_date:
        if current_date is not None:
            print()
        print(f"\n## {date_only}")
        print("-" * 100)
        current_date = date_only
    
    print(f"{date_str}  |  {filepath}")

print()
print("=" * 100)
print(f"Total files: {len(files_with_dates)}")
print("=" * 100)

