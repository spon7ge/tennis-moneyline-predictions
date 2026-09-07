from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerIdentityMap:
    version: str = "v1"
    conflicts: list[dict[str, str]] = field(default_factory=list)
    _names_by_raw_id: dict[str, str] = field(default_factory=dict, repr=False)

    def resolve(self, raw_id: str, name: str | None = None) -> str:
        canonical = str(raw_id)
        if name is not None:
            prior = self._names_by_raw_id.get(canonical)
            if prior is not None and prior != name:
                self.conflicts.append(
                    {"raw_id": canonical, "prior_name": prior, "new_name": name}
                )
            else:
                self._names_by_raw_id[canonical] = name
        return canonical
