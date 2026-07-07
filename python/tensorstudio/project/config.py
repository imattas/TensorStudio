"""Project configuration and workspace layout helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tensorstudio.typing import PathLikeStr


@dataclass(slots=True)
class ProjectConfig:
    """Small JSON-serializable project metadata container."""

    name: str
    version: str = "0.1.0"
    seed: int | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("project config requires a non-empty string name")
        version = data.get("version", "0.1.0")
        if not isinstance(version, str) or not version:
            raise ValueError("project config version must be a non-empty string")
        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("project config seed must be an int or None")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("project config metadata must be a mapping")
        return cls(name=name, version=version, seed=seed, metadata=dict(metadata))

    def save(self, path: PathLikeStr) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        output_path.write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: PathLikeStr) -> ProjectConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("project config JSON must contain an object")
        return cls.from_dict(data)


class Project:
    """Filesystem layout for a TensorStudio experiment or package project."""

    def __init__(
        self,
        root: PathLikeStr,
        config: ProjectConfig | None = None,
        *,
        create: bool = True,
    ) -> None:
        self.root = Path(root)
        self.config = config or ProjectConfig(name=self.root.name or "tensorstudio-project")
        if create:
            self.create()

    @property
    def config_path(self) -> Path:
        return self.root / "tensorstudio.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.config_path)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def checkpoint_path(self, name: str, suffix: str = ".tsnpz") -> Path:
        filename = name if name.endswith(suffix) else f"{name}{suffix}"
        return self.checkpoints_dir / filename

    def artifact_path(self, name: str) -> Path:
        return self.artifacts_dir / name

    @classmethod
    def load(cls, root: PathLikeStr) -> Project:
        project_root = Path(root)
        config = ProjectConfig.load(project_root / "tensorstudio.json")
        return cls(project_root, config=config, create=False)


__all__ = ["Project", "ProjectConfig"]
