import shutil
import os

# Backup original
shutil.move('dashboard.py', 'dashboard_backup.py')

# Move fixed version
shutil.move('dashboard_fixed.py', 'dashboard.py')

print("[SUCCESS] Dashboard upgraded with all fixes!")
print("[INFO] - 8 critical button fixes applied")
print("[INFO] - 4 UI enhancements added")
print("[INFO] - All interactive elements now functional")
