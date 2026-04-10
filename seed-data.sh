#!/bin/sh
set -e

TARGET_DIR="${1:-data}"
SEED_SOURCE_DIR="${SEED_SOURCE_DIR:-/opt/app-seed/data}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)

if [ ! -d "$SEED_SOURCE_DIR" ]; then
    if [ -d "$SCRIPT_DIR/data" ]; then
        SEED_SOURCE_DIR="$SCRIPT_DIR/data"
    elif [ -d "/app/data" ]; then
        SEED_SOURCE_DIR="/app/data"
    else
        echo "Diretorio de seed nao encontrado: $SEED_SOURCE_DIR" >&2
        exit 1
    fi
fi

mkdir -p "$TARGET_DIR"

find "$SEED_SOURCE_DIR" -type f | while read -r source_file; do
    relative_path="${source_file#$SEED_SOURCE_DIR/}"
    target_file="$TARGET_DIR/$relative_path"
    target_dir=$(dirname "$target_file")
    mkdir -p "$target_dir"
    if [ ! -f "$target_file" ]; then
        cp "$source_file" "$target_file"
    fi
done
