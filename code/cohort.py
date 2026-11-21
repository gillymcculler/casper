from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


FITTING_SITES = frozenset(('A', 'B', 'C', 'D', 'E'))
HELD_OUT_SITES = frozenset(('F', 'G'))


@dataclass(frozen=True)
class CohortSchema:
    site_column: str
    outcome_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    missing_tokens: tuple[str, ...] = ('',)

    def __post_init__(self) -> None:
        columns = (self.site_column, self.outcome_column, *self.numeric_columns, *self.categorical_columns)
        if any(not col for col in columns) or len(set(columns)) != len(columns):
            raise ValueError('Site, outcome, and feature column names must be nonempty and distinct')
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError('Select at least one feature')


@dataclass(frozen=True)
class ClinicalRecord:
    site: str
    outcome: int
    numeric: tuple[float | None, ...]
    categorical: tuple[str | None, ...]


@dataclass(frozen=True)
class RetrospectiveCohort:
    fitting: tuple[ClinicalRecord, ...]
    held_out: tuple[ClinicalRecord, ...]

    def site_counts(self) -> dict[str, int]:
        result = {site: 0 for site in sorted(FITTING_SITES | HELD_OUT_SITES)}
        for record in (*self.fitting, *self.held_out):
            result[record.site] += 1
        return result


def read_retrospective_cohort(path: str | Path, schema: CohortSchema) -> RetrospectiveCohort:
    import math

    fitting: list[ClinicalRecord] = []
    held_out: list[ClinicalRecord] = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {schema.site_column, schema.outcome_column, *schema.numeric_columns, *schema.categorical_columns}
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('The CSV must have a header with unique column names')
        if not required.issubset(reader.fieldnames):
            raise ValueError('The CSV is missing one or more configured columns')
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(name) is None for name in required):
                raise ValueError(f'Invalid field count in row {row_number}')
            site = row[schema.site_column].strip()
            if site not in FITTING_SITES | HELD_OUT_SITES:
                raise ValueError(f'Unrecognized retrospective site in row {row_number}')
            raw_outcome = row[schema.outcome_column].strip()
            if raw_outcome not in ('0', '1'):
                raise ValueError(f'Outcome must be a known binary value in row {row_number}')
            numeric: list[float | None] = []
            for column in schema.numeric_columns:
                value = row[column].strip()
                if value in schema.missing_tokens:
                    numeric.append(None)
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f'Nonfinite numeric value in row {row_number}')
                numeric.append(number)
            categorical = tuple(None if row[col].strip() in schema.missing_tokens else row[col].strip() for col in schema.categorical_columns)
            record = ClinicalRecord(site, int(raw_outcome), tuple(numeric), categorical)
            (fitting if site in FITTING_SITES else held_out).append(record)
    if not fitting or not held_out:
        raise ValueError('Both fitting and held-out records are required')
    return RetrospectiveCohort(tuple(fitting), tuple(held_out))


def cohort_summary(records: Sequence[ClinicalRecord], schema: CohortSchema) -> Mapping[str, object]:
    if not records:
        raise ValueError('At least one record is required')
    if any(len(r.numeric) != len(schema.numeric_columns) or len(r.categorical) != len(schema.categorical_columns) for r in records):
        raise ValueError('Records and schema do not match')
    numeric_missing = {name: sum(record.numeric[i] is None for record in records) for i, name in enumerate(schema.numeric_columns)}
    categorical_missing = {name: sum(record.categorical[i] is None for record in records) for i, name in enumerate(schema.categorical_columns)}
    return {'records': len(records), 'positive_outcomes': sum(r.outcome for r in records), 'missing_numeric': numeric_missing, 'missing_categorical': categorical_missing}
