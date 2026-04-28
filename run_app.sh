#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p adpcm_py/.cache adpcm_py/.matplotlib_cache
export XDG_CACHE_HOME="$PWD/adpcm_py/.cache"
export MPLCONFIGDIR="$PWD/adpcm_py/.matplotlib_cache"

if [ ! -x "venv/bin/python" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

if ! python -c "import PyQt6, numpy, matplotlib, scipy" >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

python -m adpcm_py.main "$@"
