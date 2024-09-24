#!/bin/sh


# https://www.patorjk.com/software/taag/
# Font Name: ANSI Regular
echo
echo '██████  ██████  ███████ ████████ ████████ ██    ██     ██████  ██████  ██ ███    ██ ████████'
echo '██   ██ ██   ██ ██         ██       ██     ██  ██      ██   ██ ██   ██ ██ ████   ██    ██'
echo '██████  ██████  █████      ██       ██      ████       ██████  ██████  ██ ██ ██  ██    ██'
echo '██      ██   ██ ██         ██       ██       ██        ██      ██   ██ ██ ██  ██ ██    ██'
echo '██      ██   ██ ███████    ██       ██       ██        ██      ██   ██ ██ ██   ████    ██'

MODEL="$1"
OUTPUT_FILE="arch/$(echo "$MODEL" | tr / -).md"

echo
echo "=================================================="
echo "MODEL:       $MODEL"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "=================================================="
echo

ARCH=$(uv run python -m nlnas pretty-print "$MODEL")
echo "$ARCH"

mkdir -p arch
{
    printf "# [\`$MODEL\`](https://huggingface.co/$MODEL)\n";
    echo ;
    echo '```';
    echo "$ARCH";
    echo '```';

} > "$OUTPUT_FILE"
