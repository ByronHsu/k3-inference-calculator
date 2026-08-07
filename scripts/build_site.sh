#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/_site"

rm -rf "$output_dir"
mkdir -p "$output_dir"
cp "$repo_root"/site/* "$output_dir"/
python3 "$repo_root/scripts/package_runtime.py" "$output_dir/runtime.zip"
touch "$output_dir/.nojekyll"
