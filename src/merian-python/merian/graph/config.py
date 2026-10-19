"""
Composes a graph config from command-line arguments.

Besides the graph, a config may declare:

- "include": configs merged in order before its own content, each at most once.
- "presets": {"renderer": {"folder": "renderers", "default": "pt", "order": 0}} offers
  the configs in a folder as --renderer choices; "multiple": true makes the option
  repeatable.
- "cli": {"spp": {"pointer": "/nodes/render/properties/samples per pixel"}} names a JSON
  pointer that --spp sets; "set" assigns further pointers then, "required" demands a
  value and "help" describes it. "--" takes the positional arguments; null removes an
  entry.

Presets apply by their order, the choices of one in command-line order, then --merge
configs, aliases and --set assignments.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _graph

POSITIONAL = "--"


@dataclass
class _PresetGroup:
    name: str
    choices: dict[str, Path]
    defaults: list[str]
    multiple: bool
    order: int


class Config:
    def __init__(self, path: str | Path, search_dirs: Sequence[Path] | None = None):
        dirs = list(
            _graph.default_search_dirs() if search_dirs is None else search_dirs
        )
        self.path = _graph.find_config(Path(path), dirs)
        self.search_dirs = [self.path.parent] + dirs
        self.json: dict[str, Any] = {}
        self.applied: set[Path] = set()
        self.apply(self.path)

    def preset_groups(self) -> list[_PresetGroup]:
        groups = []
        for name, spec in self.json.get("presets", {}).items():
            folder = self.find(spec["folder"])
            if folder is None or not folder.is_dir():
                raise _graph.ConfigError(
                    f"preset folder '{spec['folder']}' of --{name} not found"
                )
            defaults = spec.get("default", [])
            groups.append(
                _PresetGroup(
                    name=name,
                    choices={file.stem: file for file in sorted(folder.glob("*.json"))},
                    defaults=defaults if isinstance(defaults, list) else [defaults],
                    multiple=spec.get("multiple", False),
                    order=spec.get("order", 0),
                )
            )
        return sorted(groups, key=lambda group: (group.order, group.name))

    def merge(self, file: str | Path) -> None:
        """Merges file, searched in the working directory, then the search dirs."""
        path = Path(file) if Path(file).exists() else self.find(file)
        if path is None:
            raise _graph.ConfigError(f"config '{file}' not found")
        self.apply(path)

    def set(self, pointer: str, value) -> None:
        """Assigns value at pointer; an index one past a list's end, or "-", appends."""
        parent, key = self._parent(pointer)
        if isinstance(parent, list) and key == len(parent):
            parent.append(value)
        else:
            parent[key] = value

    def set_text(self, pointer: str, text: str) -> None:
        """Assigns text, parsed as JSON unless the value at pointer is a string."""
        value = text
        if not isinstance(self.get(pointer), str):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                pass
        self.set(pointer, value)

    def get(self, pointer: str):
        node = self.json
        for part in _pointer_parts(pointer):
            if isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    def aliases(self) -> dict[str, dict]:
        aliases = {}
        for name, spec in self.json.get("cli", {}).items():
            if spec is None:
                continue
            if "pointer" not in spec:
                raise _graph.ConfigError(f"cli entry '{name}' has no pointer")
            aliases[name] = spec
        return aliases

    def description(self) -> _graph.Description:
        config = dict(self.json)
        config["cli"] = self.aliases()
        config.pop("presets", None)
        return _graph.Description.from_dict(config)

    def find(self, file: str | Path, directory: Path | None = None) -> Path | None:
        for parent in ([directory] if directory else []) + self.search_dirs:
            if (parent / file).exists():
                return parent / file
        return None

    def apply(self, path: Path) -> None:
        """Merges the config at path, its includes first, each config at most once."""
        resolved = path.resolve()
        if resolved in self.applied:
            return
        self.applied.add(resolved)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise _graph.ConfigError(f"could not parse config '{path}': {e}") from e
        includes = config.pop("include", [])
        for file in includes if isinstance(includes, list) else [includes]:
            included = self.find(file, path.parent)
            if included is None:
                raise _graph.ConfigError(f"include '{file}' of '{path}' not found")
            self.apply(included)
        self.json = _graph.merge(self.json, config)

    def _parent(self, pointer: str) -> tuple[dict | list, str | int]:
        parts = _pointer_parts(pointer)
        node = self.json
        for part in parts[:-1]:
            if isinstance(node, list):
                node = node[_list_index(node, part, pointer, appends=False)]
            elif isinstance(node, dict):
                node = node.setdefault(part, {})
            else:
                raise _graph.ConfigError(f"{pointer}: '{part}' is inside a value")
        if isinstance(node, list):
            return node, _list_index(node, parts[-1], pointer, appends=True)
        if not isinstance(node, dict):
            raise _graph.ConfigError(f"{pointer}: '{parts[-1]}' is inside a value")
        return node, parts[-1]


def _list_index(node: list, part: str, pointer: str, appends: bool) -> int:
    end = len(node) if appends else len(node) - 1
    if part == "-" and appends:
        return len(node)
    if not part.isdigit() or int(part) > end:
        raise _graph.ConfigError(
            f"{pointer}: no element '{part}' in a list of {len(node)}"
        )
    return int(part)


class _Fragment(argparse.Action):
    """Records a preset or --merge config in command-line order."""

    def __call__(self, _parser, namespace, value, _option_string=None):
        namespace.fragments = [*getattr(namespace, "fragments", []), (self.dest, value)]


def apply_args(
    config: Config, args: Sequence[str], parser: argparse.ArgumentParser | None = None
) -> argparse.Namespace:
    """
    Applies args to config; the options of parser (made with add_help=False) are
    accepted too and returned.
    """
    base = parser or argparse.ArgumentParser(add_help=False)
    groups = config.preset_groups()

    # Presets first: their configs can declare further aliases
    presets = _parser([base])
    _add_fragment_arguments(presets, groups)
    selected, _ = parse_known(presets, args)
    given = getattr(selected, "fragments", [])
    for group in groups:
        choices = [choice for name, choice in given if name == group.name]
        if len(choices) > 1 and not group.multiple:
            raise _graph.ConfigError(f"--{group.name} given {len(choices)} times")
        for choice in choices or group.defaults:
            if choice in group.choices:
                config.apply(group.choices[choice])
            else:
                config.merge(choice)
    for name, file in given:
        if name == "merge":
            config.merge(file)

    aliases = config.aliases()
    named = {name: spec for name, spec in aliases.items() if name != POSITIONAL}
    positional = aliases.get(POSITIONAL)
    full = _parser(
        [base],
        prog=base.prog,
        description=base.description,
        epilog=f"Positional arguments set {positional['pointer']}."
        if positional
        else None,
    )
    full.add_argument("--help", action="help", help="show this help message and exit")
    _add_fragment_arguments(full, groups)
    full.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="POINTER=VALUE",
        help="assign a value at a JSON pointer, e.g. /nodes/render/properties/x=1",
    )
    for name, spec in named.items():
        try:
            full.add_argument(
                f"--{name}",
                dest=f"cli {name}",
                metavar="VALUE",
                help=spec.get("help", spec["pointer"]),
            )
        except argparse.ArgumentError as e:
            raise _graph.ConfigError(
                f"cli entry '{name}' collides with an option of that name"
            ) from e

    namespace, rest = _parse(full, args)
    separator = rest.index(POSITIONAL) if POSITIONAL in rest else len(rest)
    unknown = [token for token in rest[:separator] if token.startswith("--")]
    if unknown:
        raise _graph.ConfigError(f"unrecognized options: {' '.join(unknown)}")
    values = rest[:separator] + rest[separator + 1 :]
    if values and not positional:
        raise _graph.ConfigError(f"unrecognized arguments: {' '.join(values)}")
    if values:
        _apply_alias(config, positional, _join_quoted(values))
    for name, spec in named.items():
        value = getattr(namespace, f"cli {name}")
        if value is not None:
            _apply_alias(config, spec, value)
    for name, spec in aliases.items():
        if spec.get("required") and config.get(spec["pointer"]) in (None, ""):
            raise _graph.ConfigError(
                "a value for the positional arguments is required"
                if name == POSITIONAL
                else f"--{name} is required"
            )
    for assignment in namespace.set:
        pointer, separator, value = assignment.partition("=")
        if not separator:
            raise _graph.ConfigError(f"--set expects POINTER=VALUE, got '{assignment}'")
        config.set_text(pointer, value)
    return namespace


def load(
    path: str | Path,
    args: Sequence[str] = (),
    search_dirs: Sequence[Path] | None = None,
) -> _graph.Description:
    config = Config(path, search_dirs)
    apply_args(config, args)
    return config.description()


def _parser(
    parents: list[argparse.ArgumentParser], **kwargs
) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        parents=parents,
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        **kwargs,
    )


def parse_known(
    parser: argparse.ArgumentParser, args: Sequence[str]
) -> tuple[argparse.Namespace, list[str]]:
    """Returns the options of parser in args and the rest, required ones unchecked."""
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        return _parse(parser, args)
    finally:
        for action in required:
            action.required = True


def _parse(parser: argparse.ArgumentParser, args: Sequence[str]):
    try:
        return parser.parse_known_args(list(args))
    except argparse.ArgumentError as e:
        raise _graph.ConfigError(str(e)) from e


def _add_fragment_arguments(
    parser: argparse.ArgumentParser, groups: list[_PresetGroup]
) -> None:
    for group in groups:
        parser.add_argument(
            f"--{group.name}",
            action=_Fragment,
            dest=group.name,
            metavar="|".join(group.choices) + "|FILE.json",
            help=f"default: {', '.join(group.defaults) or 'none'}"
            + ("; repeatable" if group.multiple else ""),
        )
    parser.add_argument(
        "--merge",
        action=_Fragment,
        dest="merge",
        metavar="FILE.json",
        help="merge a config",
    )


def _apply_alias(config: Config, spec: dict, text: str) -> None:
    config.set_text(spec["pointer"], text)
    for pointer, value in spec.get("set", {}).items():
        config.set(pointer, value)


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise _graph.ConfigError(f"'{pointer}' is no JSON pointer")
    return [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]


def _join_quoted(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    quoted = []
    for value in values:
        if value and not any(c in value for c in ' \t\n\v\f\r"'):
            quoted.append(value)
        else:
            quoted.append('"' + value.replace('"', '""') + '"')
    return " ".join(quoted)
