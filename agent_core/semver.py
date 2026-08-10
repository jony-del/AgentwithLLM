"""Small Node-semver-compatible range evaluator used by plugin dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True, order=False, slots=True)
class Version:
    major: int
    minor: int = 0
    patch: int = 0
    prerelease: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "Version":
        match = re.fullmatch(
            r"[v=\s]*(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?(?:\.(0|[1-9]\d*))?"
            r"(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?\s*",
            value,
        )
        if match is None:
            raise ValueError(f"invalid semantic version: {value}")
        return cls(
            int(match.group(1)), int(match.group(2) or 0), int(match.group(3) or 0),
            tuple((match.group(4) or "").split(".")) if match.group(4) else (),
        )

    def _cmp(self, other: "Version") -> int:
        left = (self.major, self.minor, self.patch)
        right = (other.major, other.minor, other.patch)
        if left != right:
            return -1 if left < right else 1
        if not self.prerelease and not other.prerelease:
            return 0
        if not self.prerelease:
            return 1
        if not other.prerelease:
            return -1
        for left_item, right_item in zip(self.prerelease, other.prerelease):
            if left_item == right_item:
                continue
            left_num, right_num = left_item.isdigit(), right_item.isdigit()
            if left_num and right_num:
                return -1 if int(left_item) < int(right_item) else 1
            if left_num != right_num:
                return -1 if left_num else 1
            return -1 if left_item < right_item else 1
        return (len(self.prerelease) > len(other.prerelease)) - (len(self.prerelease) < len(other.prerelease))

    def __lt__(self, other: "Version") -> bool:
        return self._cmp(other) < 0


def _compare(version: Version, operator: str, target: Version) -> bool:
    value = version._cmp(target)
    return {
        "": value == 0, "=": value == 0, "==": value == 0,
        ">": value > 0, ">=": value >= 0, "<": value < 0, "<=": value <= 0,
    }[operator]


def _upper_caret(version: Version) -> Version:
    if version.major:
        return Version(version.major + 1)
    if version.minor:
        return Version(0, version.minor + 1)
    return Version(0, 0, version.patch + 1)


def _test_set(version: Version, expression: str) -> bool:
    expression = expression.strip()
    if not expression or expression in {"*", "latest"}:
        return not version.prerelease
    hyphen = re.fullmatch(r"\s*(\S+)\s+-\s+(\S+)\s*", expression)
    if hyphen:
        return _compare(version, ">=", Version.parse(hyphen.group(1))) and _compare(
            version, "<=", Version.parse(hyphen.group(2))
        )
    tokens = expression.replace(",", " ").split()
    allow_prerelease = any("-" in token for token in tokens)
    if version.prerelease and not allow_prerelease:
        return False
    for token in tokens:
        match = re.fullmatch(r"(<=|>=|<|>|=|==|\^|~)?\s*([vV]?[^\s]+)", token)
        if match is None:
            raise ValueError(f"invalid semver comparator: {token}")
        operator, raw = match.group(1) or "", match.group(2).lstrip("vV")
        if any(char in raw for char in "xX*"):
            parts = raw.split(".")
            concrete = [part for part in parts if part not in {"x", "X", "*"}]
            lower = Version.parse(".".join(concrete) if concrete else "0")
            if not _compare(version, ">=", lower):
                return False
            if not concrete:
                continue
            upper = Version(lower.major + 1) if len(concrete) == 1 else Version(lower.major, lower.minor + 1)
            if not _compare(version, "<", upper):
                return False
            continue
        target = Version.parse(raw)
        if operator == "^":
            if not (_compare(version, ">=", target) and _compare(version, "<", _upper_caret(target))):
                return False
        elif operator == "~":
            pieces = raw.split("-", 1)[0].split(".")
            upper = Version(target.major + 1) if len(pieces) == 1 else Version(target.major, target.minor + 1)
            if not (_compare(version, ">=", target) and _compare(version, "<", upper)):
                return False
        elif not _compare(version, operator, target):
            return False
    return True


def satisfies(version: str | Version, range_expression: str) -> bool:
    parsed = version if isinstance(version, Version) else Version.parse(version)
    return any(_test_set(parsed, item) for item in range_expression.split("||"))


def select_highest(versions: Iterable[str], ranges: Iterable[str]) -> str | None:
    constraints = tuple(item for item in ranges if item.strip())
    candidates: list[tuple[Version, str]] = []
    for raw in versions:
        try:
            parsed = Version.parse(raw)
            if all(satisfies(parsed, constraint) for constraint in constraints):
                candidates.append((parsed, raw))
        except ValueError:
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
