# Provenance Text Format

The toolkit writes both JSON and annotated text provenance files.

## Purpose

The `.json` file is the canonical machine record.

The `.txt` file is intended to be comfortable for a scientist to read while
remaining deterministic enough for scripts to parse.

## Grammar

- UTF-8 text.
- Blank lines are ignored.
- Lines beginning with `#` are comments.
- `[section_name]` starts a section.
- Data lines use:

```text
key = JSON_VALUE
```

`JSON_VALUE` follows standard JSON syntax:

```text
true
false
null
3072
1.25
"detector_row"
["frame_001.tif","frame_002.tif"]
{"key":"value"}
```

This means the file keeps data types rather than turning every value into a
free-form string.

## Reference parser

```python
from xrd_toolkit.provenance import read_provenance_text

record = read_provenance_text("sample_corrected.provenance.txt")

assert record["product"]["array_rows"] == 3072
assert record["coordinate_system"]["rotation_applied"] is False
```

## Why comments are useful

Each field is preceded by one or more comments explaining what it means for
later XRFD analysis. A parser may ignore those comments, while a researcher can
use them as an embedded data dictionary.

## Pairing with an exported TIFF

When automatic sidecars are enabled:

```text
sample_corrected.tif
sample_corrected.provenance.json
sample_corrected.provenance.txt
```

The provenance includes a SHA-256 checksum of the TIFF so a later script can
verify that the sidecar still corresponds to the same binary file.
