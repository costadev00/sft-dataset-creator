# dataset-unchunker

Generic utility for reconstructing datasets that were split into chunk rows.

The package is intentionally isolated from `sft_dataset_creator`. It can read a
local Parquet file, a Parquet file from a Hugging Face dataset repo, or regular
Hub dataset splits, then write a dechunked dataset with reports.

## Example

```bash
cd tools/dataset_unchunker
python -m dataset_unchunker unchunk \
  --repo-id costadev00/stf-acordaos-cpt-2048 \
  --revision b095e08fb6f0a680e359edce815f5761e4a25d52 \
  --source-file data/cpt_mixed_2048.parquet \
  --text-column text \
  --chunk-index-column chunk_index \
  --chunk-total-column chunk_total \
  --group-by source_oid,numero,label,votante \
  --tokenizer Qwen/Qwen3-8B \
  --max-groups 50 \
  --no-push
```

Set `HF_TOKEN` when reading gated datasets or publishing to the Hub.
