"""
This built-hook sets up symlinks to the different notebook-directories, i. e.
./analyses and ./data-preprocessing.
"""

from pathlib import Path
import os


def construct_symlink(source: str, target_parent: Path):
    """A wrapper to construct a symlink with sane defaults. The source is
    constructed as a full global path, to work properly.

    Parameters
    ----------
    source : str
        Name of the directory (in the current working directory) to be mapped
        symlinked to target.
    target_parent : Path
        Path to the target (parent) directory of the symlink. The symlink itself
        gets the same name as the source.
    """
    source_path = str(Path(os.getcwd()) / source)
    target = str(target_parent / source)
    os.symlink(source_path, target, target_is_directory=True)




# Main -------------------------------------------------------------------------
target_parent = Path("./mkdocs/content/autobuilt/notebooks-symlinks/")
target_parent.mkdir(exist_ok=True)

construct_symlink("data-preprocessing", target_parent)
construct_symlink("analyses", target_parent)

print("Set up notebooks-symlinks.")
