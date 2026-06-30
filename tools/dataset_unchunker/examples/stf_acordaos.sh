#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m dataset_unchunker unchunk \
  --repo-id costadev00/stf-acordaos-cpt-2048 \
  --revision b095e08fb6f0a680e359edce815f5761e4a25d52 \
  --source-file data/cpt_mixed_2048.parquet \
  --text-column text \
  --chunk-index-column chunk_index \
  --chunk-total-column chunk_total \
  --group-by source_oid,numero,label,votante \
  --strategy text-overlap \
  --tokenizer Qwen/Qwen3-8B \
  --max-overlap 2000 \
  --output-dir output/stf-acordaos-unchunked \
  "$@"
