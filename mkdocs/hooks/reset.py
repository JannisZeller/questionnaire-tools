"""
This build-hook resets all autbobuilt contents of the MkDocs site including
    - Full-reference markdowns created by ./mkdocs/hooks/full-reference-autobuild.py
    - Notebook symlinks to ./analyses and ./data-preprocessing created by
        ./mkdocs/hooks/notebook-symlinks.py
    - Reports symlinks to ./reports created by ./mkdocs/hooks/reports-symlinks.py

It is taken care, that no directories or files inside the symlinked folders
get deleted by unlinking the symlinks first.
"""

from pathlib import Path


def unlink_dir(directory: Path):
    """Convenience wrapper for unlinking a directory, i. e. dropping a symlink.

    Parameters
    ----------
    directory : Path
        A `pathlib.Path` to the directory to be unlinked.
    """
    try:
        directory.unlink()
        # print(f"Unlinked {directory}")
    except PermissionError:
        print(
            f"`PermissionError` when trying to unlink {directory}. " +
            "Perhaps it is a directory and not a symlink."
        )
        raise PermissionError
    except FileNotFoundError:
        pass


def remove_pth(path: Path):
    """Convenience wrapper for removing a file or directory.

    Parameters
    ----------
    path : Path
        A `pathlib.Path` to the file / directory to be removed.
    """
    try:
        path.rmdir()
    except FileNotFoundError:
        print(
            f"Directory {path} not found. Nothing to remove."
        )


def unlink_subdirs(directory: Path):
    """Drops symlinks inside the specified directory. Meant primarily for the
    2 notebook-symlinks (analyses and data-preprocessing)

    Parameters
    ----------
    directory : Path
        A `pathlib.Path` to the directory containing the multiple symlinks.
        Nothing else than symlinks should be contained, otherwise a
        Perission error will be raised.
    """
    try:
        for subdir in directory.iterdir():
            unlink_dir(subdir)
    except FileNotFoundError:
        pass


def remove_dir(directory: Path):
    """Recursively removing a directory including all its contents. Source:
    https://stackoverflow.com/a/49782093

    Parameters
    ----------
    directory : Path
        A `pathlib.Path` to the directory to be recursively removed.
    """
    for item in directory.iterdir():
        if item.is_dir():
            remove_dir(item)
        else:
            item.unlink()
    remove_pth(directory)


def empty_autobuilt(autobuilt_dir: Path = Path("./mkdocs/content/autobuilt")):
    """Wrapper for emptying the ./mkdocs/content/autobuilt.

    Parameters
    ----------
    autobuilt_dir : Path = Path("./mkdocs/content/autobuilt"))
        `pathlib.Path` specifying the location of the autobuilt contents
    """
    if not autobuilt_dir.is_dir():
        return
    unlink_subdirs(autobuilt_dir / "notebooks-symlinks")
    unlink_dir(autobuilt_dir / "reports-symlink")
    remove_dir(autobuilt_dir)
    autobuilt_dir.mkdir()




# Main -------------------------------------------------------------------------
empty_autobuilt()
print("Resetted autobuilt MkDocs contents.")
