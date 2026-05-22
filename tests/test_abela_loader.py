from pathlib import Path

from progen.abela import LINKER, load_abela_records, split_serialized_antibody


def test_load_abela_minimal_example():
    repo_root = Path(__file__).resolve().parents[1]
    records = load_abela_records(
        repo_root / "examples" / "abela_minimal.csv",
        active_only=True,
    )

    assert len(records) == 2
    assert records[0].target == "SyntheticTarget"
    assert records[0].row == {}
    assert records[0].serialized.startswith("1")
    assert records[0].serialized.endswith("2")
    assert records[0].serialized.count(LINKER) == 2
    assert len(records[0].hcdr_ranges) == 3

    parts = split_serialized_antibody(records[0].serialized)
    assert parts["epitope"] == records[0].epitope
    assert parts["vh"] == records[0].vh
    assert parts["vl"] == records[0].vl


def test_load_abela_keeps_raw_row_only_when_requested():
    repo_root = Path(__file__).resolve().parents[1]
    records = load_abela_records(
        repo_root / "examples" / "abela_minimal.csv",
        active_only=True,
        include_raw_row=True,
    )

    assert "ID" in records[0].row


def test_load_abela_skips_invalid_rows_by_default(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "ID,ELISA_status,H,HCDR1,HCDR2,HCDR3,L,Antigen,Antigen_sequence\n"
        "bad,AbELA-Q,,AAA,BBB,CCC,ACDE,Target,ACDE\n"
    )

    assert load_abela_records(path, active_only=False) == []
