"""
Scan all scripts for garbled Unicode characters from encoding issues
and fix them generically.
"""
import re
import os
import glob

# Common garbled Unicode patterns and their replacements
replacements = [
    # Em-dash (—) garbled as UTF-8 bytes misread as Latin-1
    (r'Ã¢â‚¬â€[œ]', '"'),
    (r'Ã¢â‚¬â„¢', "'"),
    (r'Ã¢â‚¬\x9c', '"'),
    (r'â€"', '—'),
    (r'â€™', "'"),
    (r'â€œ', '"'),
    (r'â€\x9d', '"'),
    # Generic: any long garbled pattern between 2010 and 2024
    (r'2010[ÃƒÆ\-â‚¬â€œÂ¢Â¬Â¦Ã]+2024', '2010-2024'),
]

scripts = glob.glob('scripts/*.py')
total_fixed = 0

for script_path in sorted(scripts):
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {os.path.basename(script_path)}")
        total_fixed += 1

print(f"\nTotal scripts fixed: {total_fixed}")
print("Done!")
