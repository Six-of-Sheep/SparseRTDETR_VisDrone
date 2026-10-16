"""Fail-closed parser for the eight-field VisDrone annotation line."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


class AnnotationParseError(ValueError):
    """Raised for malformed or ambiguous annotation input."""


_INTEGER = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class AnnotationFields:
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int
    truncation: float
    occlusion: float

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float | int, ...]:
        return (self.left, self.top, self.width, self.height, self.score, self.category, self.truncation, self.occlusion)


@dataclass(frozen=True)
class ParsedAnnotation:
    physical_line_number: int
    raw_line_sha256: str
    raw_fields: tuple[str, ...]
    fields: AnnotationFields
    had_single_trailing_empty_field: bool


def _line_body(raw_line: bytes) -> bytes:
    if not isinstance(raw_line, bytes):
        raise AnnotationParseError("annotation line must be bytes")
    return raw_line.rstrip(b"\r\n")


def _finite_float(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationParseError(f"{field_name} is not numeric") from exc
    if not math.isfinite(parsed):
        raise AnnotationParseError(f"{field_name} must be finite")
    return parsed


def _category(value: str) -> int:
    if not _INTEGER.fullmatch(value):
        raise AnnotationParseError("category must be an integer field")
    return int(value)


def parse_annotation_line(raw_line: bytes, physical_line_number: int) -> ParsedAnnotation:
    """Parse one physical line while retaining its byte-level identity."""

    if physical_line_number < 1:
        raise AnnotationParseError("physical line number must be positive")
    body = _line_body(raw_line)
    if not body:
        raise AnnotationParseError("blank annotation line")
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnnotationParseError("annotation line is not UTF-8") from exc
    fields = decoded.split(",")
    had_trailing = bool(fields and fields[-1] == "")
    if had_trailing:
        fields = fields[:-1]
    if len(fields) != 8:
        raise AnnotationParseError("annotation line must have exactly eight fields")
    if any(field == "" for field in fields):
        raise AnnotationParseError("annotation fields must not be empty")
    parsed = AnnotationFields(
        left=_finite_float(fields[0], "left"),
        top=_finite_float(fields[1], "top"),
        width=_finite_float(fields[2], "width"),
        height=_finite_float(fields[3], "height"),
        score=_finite_float(fields[4], "score"),
        category=_category(fields[5]),
        truncation=_finite_float(fields[6], "truncation"),
        occlusion=_finite_float(fields[7], "occlusion"),
    )
    return ParsedAnnotation(
        physical_line_number=physical_line_number,
        raw_line_sha256=hashlib.sha256(raw_line).hexdigest(),
        raw_fields=tuple(fields),
        fields=parsed,
        had_single_trailing_empty_field=had_trailing,
    )


def parse_annotation_bytes(annotation_bytes: bytes) -> tuple[ParsedAnnotation, ...]:
    """Parse annotation bytes in physical order without filesystem access."""

    if not isinstance(annotation_bytes, bytes):
        raise AnnotationParseError("annotation input must be bytes")
    return tuple(
        parse_annotation_line(line, line_number)
        for line_number, line in enumerate(annotation_bytes.splitlines(keepends=True), 1)
    )
