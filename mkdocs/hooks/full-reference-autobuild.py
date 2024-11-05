"""
This build-hook generates the markdown files for the `qtools` python-modules.
It sets up a module structure and populates it with markdown-files pointing
to the specific references.
"""

from pathlib import Path
from os import sep as pathsep


def exclude_pycache(paths: list[Path]) -> list[Path]:
    """Drops all paths containing `"__pycache__"` from a list of paths.
    """
    paths = [p for p in paths if not str(p).__contains__("__pycache__")]
    return paths


def exclude_inits(paths: list[Path]) -> list[Path]:
    """Drops all paths containing `"__init__"` from a list of paths.
    """
    files = [p for p in paths if not str(p).__contains__("__init__")]
    return files


def create_module_structure(path: Path, target_path: Path):
    """Creates a MkDocs module structure for a specified path containing a
    Python (sub-)module. The path should therefore point to a directory.

    Parameters
    ----------
    path : Path
        Path to the Python module's directory.
    target_path : Path
        Target path for the autobuilt documentation.
    """
    new_path = target_path / path.name
    new_path.mkdir(parents=True, exist_ok=True)
    new_file = new_path.with_suffix(".md")
    with new_file.open("w", encoding="utf-8") as f:
        src_module = str(path).replace(pathsep, ".")
        f.write(f'::: {src_module}')
        print(f"Module structure created for {src_module}")


def create_pyfile_structure(path: Path, target_path: Path):
    """Creates a MkDocs markdown file for a specified path containing a
    Python (sub-)modules `.py`-file.

    Parameters
    ----------
    path : Path
        Path to the Python module's file.
    target_path : Path
        Target path for the autobuilt documentation.
    """
    new_file = Path(*path.parent.parts[1:]) / path.with_suffix(".md").name
    new_file = target_path / new_file
    with new_file.open("w", encoding="utf-8") as f:
        src_file = str(path).replace(".py", "")
        src_file = str(src_file).replace(pathsep, ".")
        f.write(f'::: {src_file}')
        # print(f"File structure created for {src_file}")


def create_mkdocs_for_path(path: Path, target_path: Path):
    """A wrapper that checks whether the path passed is a directory or a file
    and passing it on to the correct creation-function.
    """
    if path.is_dir():
        create_module_structure(path, target_path)
    else:
        create_pyfile_structure(path, target_path)


def autobuild_reference(
    base_path: Path=Path("./qutools"),
    target_path: Path=Path("./mkdocs/content/autobuilt")
):
    """A Wrapper for automatically building a MkDocs full reference
    Makdown-structure for the module and submodlues in `base_path` and store it
    to `target_path`.
    """
    if not target_path.is_dir():
        target_path.mkdir()
    src_paths = base_path.rglob("*")
    src_paths = exclude_pycache(src_paths)
    src_paths = exclude_inits(src_paths)
    for path in src_paths:
        if "_toydata" in str(path):
            continue
        create_mkdocs_for_path(path, target_path)




# Main -------------------------------------------------------------------------
autobuild_reference()
print("Created MkDocs full reference contents.")
