"""
# Submodule with unique-key yaml-loader
"""

from pathlib import Path
import yaml



class UniqueKeyError(Exception):
    """An Exception-class for the `UniqueKeyLoader` yaml-loader class.
    """
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    """A yaml-loader class, that asserts unique keys on all levels of the yaml
    file. Credits to: [this comment](https://github.com/AppDaemon/appdaemon/issues/1899#issuecomment-1858683299).
    """
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            if ':merge' in key_node.tag:
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise UniqueKeyError(f"Duplicate {key!r} task-name (top-level key) found in config-yaml.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def load_unique_key_yaml(path: Path) -> dict:
    """A wrapper for loading a unique-key yaml-file. Catches errors and prints
    them as warnings.

    Parameters
    ----------
    path : Path
        Path to a `.yaml` or `.yml` file.

    Returns
    -------
    dict
        The contents of the yaml-file as a python dictionary.
    """
    with open(path, "r") as file:
        try:
            config_dict = yaml.load(file, Loader=UniqueKeyLoader)
            return config_dict
        except yaml.YAMLError as exc:
            print(f"WARNING: Error when trying to load {path}: {exc}")
