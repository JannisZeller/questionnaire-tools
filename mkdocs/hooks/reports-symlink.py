"""
This built-hook sets up symlinks to the reports-directory.
"""

from pathlib import Path
import os




# Main -------------------------------------------------------------------------
source = str(Path(os.getcwd()) / "reports")
target = "./mkdocs/content/autobuilt/reports-symlink"
os.symlink(source, target, target_is_directory=True)
