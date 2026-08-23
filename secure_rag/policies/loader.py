import re
from pathlib import Path
from typing import Union

import yaml

from .engine import DECISION_KEEP, DECISION_MASK, ConfigurablePolicy

_BUILTIN_DIR = Path(__file__).resolve().parent.parent / "policy_configs"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_YAML_SUFFIXES = {".yaml", ".yml"}
_ALLOWED_TOP_LEVEL = {"version", "name", "description", "entities", "preserve"}
_ALLOWED_PRESERVE = {"categories", "terms"}
_ALLOWED_DECISIONS = {DECISION_MASK, DECISION_KEEP}
_SUPPORTED_VERSION = 1


class PolicyError(Exception):
    """Base class for policy loading/configuration failures."""


class PolicyNotFoundError(PolicyError):
    """A policy name or path could not be resolved to a YAML file."""


class PolicyConfigError(PolicyError):
    """A policy YAML file exists but its configuration is invalid."""


def load_policy(source: Union[str, Path]) -> ConfigurablePolicy:
    """Load a policy from a builtin name ("default", "medical", ...) or a
    path to a YAML file, and return a ready-to-use policy engine.

    YAML -> validated configuration -> ConfigurablePolicy.
    """
    path = _resolve_path(source)
    data = _read_yaml(path)
    config = _validate(data, path)
    preserve = config.get("preserve") or {}
    return ConfigurablePolicy(
        entity_decisions=config.get("entities") or {},
        preserved_terms=preserve.get("terms") or (),
        preserved_categories=preserve.get("categories") or (),
        name=str(config.get("name") or path.stem),
    )


def _resolve_path(source: Union[str, Path]) -> Path:
    source = str(source)
    path = Path(source)
    looks_like_path = path.suffix.lower() in _YAML_SUFFIXES
    if looks_like_path:
        if not path.is_file():
            raise PolicyNotFoundError(f"policy file not found: {source}")
        return path
    if _NAME_PATTERN.match(source):
        builtin = _BUILTIN_DIR / f"{source}.yaml"
        if not builtin.is_file():
            available = sorted(p.stem for p in _BUILTIN_DIR.glob("*.yaml"))
            raise PolicyNotFoundError(
                f"unknown policy {source!r}; no such builtin in {_BUILTIN_DIR} "
                f"(available: {', '.join(available) or 'none'}) and "
                f"'{source}' is not a path to an existing .yaml file"
            )
        return builtin
    if path.is_file():
        return path
    raise PolicyNotFoundError(
        f"invalid policy source {source!r}: not a builtin policy name and not a path to a YAML file"
    )


def _read_yaml(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise PolicyConfigError(f"{path}: malformed YAML: {exc}") from exc
    except OSError as exc:
        raise PolicyNotFoundError(f"{path}: cannot read policy file: {exc}") from exc
    return data


def _validate(data, path: Path) -> dict:
    def fail(message: str):
        raise PolicyConfigError(f"{path}: invalid policy configuration: {message}")

    if data is None:
        fail("file is empty")
    if not isinstance(data, dict):
        fail(f"expected a mapping at the top level, got {type(data).__name__}")

    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        fail(f"unknown top-level key(s): {sorted(unknown)}; allowed: {sorted(_ALLOWED_TOP_LEVEL)}")

    version = data.get("version", _SUPPORTED_VERSION)
    if version != _SUPPORTED_VERSION:
        fail(f"unsupported version {version!r}; expected {_SUPPORTED_VERSION}")

    for key in ("name", "description"):
        if key in data and not isinstance(data[key], str):
            fail(f"'{key}' must be a string")

    entities = data.get("entities")
    if entities is not None:
        if not isinstance(entities, dict):
            fail("'entities' must be a mapping of label -> decision")
        for label, decision in entities.items():
            if not isinstance(label, str) or not label.strip():
                fail(f"'entities' keys must be non-empty strings, got {label!r}")
            if not isinstance(decision, str) or decision.lower() not in _ALLOWED_DECISIONS:
                fail(
                    f"'entities.{label}' must be one of {sorted(_ALLOWED_DECISIONS)}, "
                    f"got {decision!r}"
                )

    preserve = data.get("preserve")
    if preserve is not None:
        if not isinstance(preserve, dict):
            fail("'preserve' must be a mapping with optional 'categories' and 'terms'")
        unknown_preserve = set(preserve) - _ALLOWED_PRESERVE
        if unknown_preserve:
            fail(f"unknown 'preserve' key(s): {sorted(unknown_preserve)}; allowed: {sorted(_ALLOWED_PRESERVE)}")
        for key in ("categories", "terms"):
            values = preserve.get(key)
            if values is None:
                continue
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                fail(f"'preserve.{key}' must be a list of non-empty strings")
    return data
