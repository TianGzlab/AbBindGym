import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


START_TOKEN = "1"
END_TOKEN = "2"
LINKER = "GGGGSGGGGSGGGGS"
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROGEN_AMINO_ACIDS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

DEFAULT_ACTIVE_VALUES = ("AbELA-Q", "active", "positive", "binder", "true", "1", "yes")
EPITOPE_ALIASES = ("epitope", "epitope_region", "epitope_sequence", "antigen_epitope", "antigen_sequence")
VH_ALIASES = ("vh", "VH", "h", "H", "heavy", "heavy_chain", "heavy_sequence", "heavy_chain_variable", "heavy_variable")
VL_ALIASES = ("vl", "VL", "l", "L", "light", "light_chain", "light_sequence", "light_chain_variable", "light_variable")
ID_ALIASES = ("id", "record_id", "name", "sequence_id", "antibody_id")
TARGET_ALIASES = ("target", "antigen", "antigen_target", "target_name")
ACTIVE_ALIASES = (
    "subset",
    "dataset",
    "activity",
    "active",
    "is_active",
    "label",
    "status",
    "elisa_status",
    "ELISA_status",
    "split",
)
EC50_ALIASES = ("EC50(ng/ML)", "ec50", "ec50_ng_ml", "ec50_ng_per_ml")


@dataclass(frozen=True)
class AbelaRecord:
    record_id: str
    target: str
    epitope: str
    vh: str
    vl: str
    hcdr_ranges: tuple
    row: dict

    @property
    def serialized(self):
        return serialize_antibody(self.epitope, self.vh, self.vl)


def normalize_sequence(value, field_name):
    seq = re.sub(r"[\s\-_.]", "", str(value or "")).upper()
    if not seq:
        raise ValueError(f"{field_name} is empty")

    invalid = sorted(set(seq) - set(PROGEN_AMINO_ACIDS))
    if invalid:
        raise ValueError(f"{field_name} contains tokens outside the ProGen2 vocabulary: {invalid}")
    return seq


def serialize_antibody(epitope, vh, vl, linker=LINKER):
    epitope = normalize_sequence(epitope, "epitope")
    vh = normalize_sequence(vh, "vh")
    vl = normalize_sequence(vl, "vl")
    return f"{START_TOKEN}{epitope}{linker}{vh}{linker}{vl}{END_TOKEN}"


def split_serialized_antibody(serialized, linker=LINKER):
    value = str(serialized).strip()
    if not value.startswith(START_TOKEN) or not value.endswith(END_TOKEN):
        raise ValueError("serialized antibody must start with '1' and end with '2'")

    body = value[1:-1]
    parts = body.split(linker)
    if len(parts) != 3:
        raise ValueError("serialized antibody must contain exactly two linker delimiters")
    epitope, vh, vl = parts
    return {
        "epitope": normalize_sequence(epitope, "epitope"),
        "vh": normalize_sequence(vh, "vh"),
        "vl": normalize_sequence(vl, "vl"),
    }


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        with path.open() as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("records", "data", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"cannot find a record list in {path}")

    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel_tab if suffix == ".tsv" else csv.excel
        return list(csv.DictReader(handle, dialect=dialect))


def find_column(row, preferred, aliases):
    if preferred:
        if preferred not in row:
            raise KeyError(f"column '{preferred}' was not found")
        return preferred

    lowered = {key.lower(): key for key in row.keys()}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def parse_boolish_set(values):
    if values is None:
        return {value.lower() for value in DEFAULT_ACTIVE_VALUES}
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    return {str(value).lower() for value in values}


def load_abela_records(
    path,
    epitope_column=None,
    vh_column=None,
    vl_column=None,
    id_column=None,
    target_column=None,
    active_column=None,
    active_only=True,
    active_values=None,
    ec50_column=None,
    max_ec50=None,
    cdr_index_base=1,
    skip_invalid=True,
    include_raw_row=False,
):
    rows = read_table(path)
    if not rows:
        return []

    first = rows[0]
    epitope_column = find_column(first, epitope_column, EPITOPE_ALIASES)
    vh_column = find_column(first, vh_column, VH_ALIASES)
    vl_column = find_column(first, vl_column, VL_ALIASES)
    id_column = find_column(first, id_column, ID_ALIASES)
    target_column = find_column(first, target_column, TARGET_ALIASES)

    if active_only:
        active_column = find_column(first, active_column, ACTIVE_ALIASES)

    if max_ec50 is not None:
        ec50_column = find_column(first, ec50_column, EC50_ALIASES)
        if ec50_column is None:
            raise KeyError("max_ec50 was provided, but no EC50 column was found")

    missing = [
        name
        for name, value in (("epitope", epitope_column), ("vh", vh_column), ("vl", vl_column))
        if value is None
    ]
    if missing:
        raise KeyError(f"missing required AbELA column(s): {', '.join(missing)}")

    active_value_set = parse_boolish_set(active_values)
    records = []
    for idx, row in enumerate(rows):
        try:
            if not row_passes_filters(
                row,
                active_column=active_column,
                active_only=active_only,
                active_value_set=active_value_set,
                ec50_column=ec50_column,
                max_ec50=max_ec50,
            ):
                continue
            record = make_abela_record(
                row,
                index=idx,
                epitope_column=epitope_column,
                vh_column=vh_column,
                vl_column=vl_column,
                id_column=id_column,
                target_column=target_column,
                cdr_index_base=cdr_index_base,
                include_raw_row=include_raw_row,
            )
        except ValueError:
            if skip_invalid:
                continue
            raise
        records.append(record)

    return records


def row_passes_filters(row, active_column, active_only, active_value_set, ec50_column, max_ec50):
    if active_only and active_column is not None:
        value = str(row.get(active_column, "")).strip().lower()
        if value not in active_value_set:
            return False

    if max_ec50 is not None:
        value = str(row.get(ec50_column, "")).strip()
        if not value:
            return False
        if float(value) > float(max_ec50):
            return False

    return True


def make_abela_record(
    row,
    index,
    epitope_column,
    vh_column,
    vl_column,
    id_column,
    target_column,
    cdr_index_base,
    include_raw_row=False,
):
    record_id = str(row.get(id_column, "")).strip() if id_column else ""
    if not record_id:
        record_id = f"record_{index:05d}"

    target = str(row.get(target_column, "")).strip() if target_column else ""
    epitope = normalize_sequence(row.get(epitope_column), "epitope")
    vh = normalize_sequence(row.get(vh_column), "vh")
    vl = normalize_sequence(row.get(vl_column), "vl")
    hcdr_ranges = parse_hcdr_ranges(row, vh_length=len(vh), index_base=cdr_index_base)
    return AbelaRecord(
        record_id=record_id,
        target=target,
        epitope=epitope,
        vh=vh,
        vl=vl,
        hcdr_ranges=hcdr_ranges,
        row=dict(row) if include_raw_row else {},
    )


def parse_hcdr_ranges(row, vh_length=None, index_base=1):
    range_column = find_column(
        row,
        None,
        ("hcdr_ranges", "hcdrs", "cdrh_ranges", "heavy_cdr_ranges", "h_cdr_ranges"),
    )
    if range_column:
        ranges = parse_range_list(row.get(range_column))
        return validate_ranges(ranges, vh_length=vh_length, index_base=index_base)

    ranges = []
    for cdr in ("1", "2", "3"):
        single_column = find_column(row, None, (f"hcdr{cdr}", f"h_cdr{cdr}", f"cdrh{cdr}", f"cdr_h{cdr}"))
        if single_column and row.get(single_column):
            value = str(row.get(single_column)).strip()
            if re.search(r"\d", value):
                ranges.append(parse_single_range(value))
            continue

        start_column = find_column(
            row,
            None,
            (
                f"hcdr{cdr}_start",
                f"h_cdr{cdr}_start",
                f"cdrh{cdr}_start",
                f"cdr_h{cdr}_start",
                f"heavy_cdr{cdr}_start",
            ),
        )
        end_column = find_column(
            row,
            None,
            (
                f"hcdr{cdr}_end",
                f"h_cdr{cdr}_end",
                f"cdrh{cdr}_end",
                f"cdr_h{cdr}_end",
                f"heavy_cdr{cdr}_end",
            ),
        )
        if start_column and end_column and row.get(start_column) and row.get(end_column):
            ranges.append((int(row[start_column]), int(row[end_column])))

    if ranges:
        return validate_ranges(ranges, vh_length=vh_length, index_base=index_base)

    vh_column = find_column(row, None, VH_ALIASES)
    if not vh_column:
        return ()

    vh = normalize_sequence(row.get(vh_column), "vh")
    cdr_sequence_ranges = []
    search_start = 0
    for cdr in ("1", "2", "3"):
        single_column = find_column(row, None, (f"hcdr{cdr}", f"h_cdr{cdr}", f"cdrh{cdr}", f"cdr_h{cdr}"))
        if not single_column or not row.get(single_column):
            continue
        cdr_seq = normalize_sequence(row.get(single_column), single_column)
        start = vh.find(cdr_seq, search_start)
        if start < 0:
            start = vh.find(cdr_seq)
        if start < 0:
            raise ValueError(f"{single_column} sequence was not found in VH")
        end = start + len(cdr_seq)
        cdr_sequence_ranges.append((start, end))
        search_start = end
    return tuple(cdr_sequence_ranges)


def parse_range_list(value):
    if value is None or str(value).strip() == "":
        return []

    if isinstance(value, (list, tuple)):
        return [tuple(item) for item in value]

    text = str(value).strip()
    if text.startswith("["):
        data = json.loads(text)
        return [tuple(item) for item in data]

    return [parse_single_range(part) for part in re.split(r"[;|,]", text) if part.strip()]


def parse_single_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    text = str(value).strip()
    match = re.match(r"^\s*(\d+)\s*(?:-|:|\.\.)\s*(\d+)\s*$", text)
    if not match:
        raise ValueError(f"cannot parse CDR range: {value!r}")
    return int(match.group(1)), int(match.group(2))


def validate_ranges(ranges, vh_length=None, index_base=1):
    normalized = []
    for start, end in ranges:
        start = int(start) - index_base
        end = int(end) - index_base + 1
        if start < 0 or end <= start:
            raise ValueError(f"invalid CDR range after index-base conversion: {(start, end)}")
        if vh_length is not None and end > vh_length:
            raise ValueError(f"CDR range {(start, end)} exceeds VH length {vh_length}")
        normalized.append((start, end))
    return tuple(normalized)


def cdr_positions(ranges):
    positions = []
    for start, end in ranges:
        positions.extend(range(start, end))
    return tuple(dict.fromkeys(positions))


def format_mutation(position, old, new, index_base=1):
    return f"H{position + index_base}{old}>{new}"
