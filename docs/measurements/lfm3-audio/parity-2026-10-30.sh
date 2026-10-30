#!/bin/zsh
# LFM3-Audio input vs llama.cpp: encoder rows, then greedy text through ferrum-cli and llama-mtmd-cli.
# Run from the repo root at the branch commit:
#   cargo build --release --bin ferrum-cli --example hybrid_audio_check
# First run at bb1e87b; the record is from the rerun after the review fixes (see README.md).
#   caffeinate -dims zsh docs/measurements/lfm3-audio/parity-2026-10-30.sh > docs/measurements/lfm3-audio/parity-2026-10-30.txt 2>&1
#
# Setup it expects in S (it creates none of it):
#   S/llama.cpp: llama.cpp at 792fcf4, cmake -B build && cmake --build build -j (llama-mtmd-cli, libmtmd)
#   S/embdump/embdump: embdump.cpp from this folder, built against that tree, e.g.
#     clang++ -std=c++17 -O2 embdump.cpp -I$L/include -I$L/ggml/include -I$L/tools/mtmd \
#       -L$L/build/bin -lllama -lmtmd -lggml -lggml-base -Wl,-rpath,$L/build/bin -o embdump
#     (L = the llama.cpp tree; it exits with an assertion after writing, since it frees nothing)
#   S/audio/*.wav, made with macOS say (the voice is the machine's default, so the clips are not committed):
#     say -o fox.wav --data-format=LEI16@16000 "The quick brown fox jumps over the lazy dog."
#     say -o weather.wav --data-format=LEI16@16000 "Tomorrow will be mostly sunny in Chicago, with a high of seventy two degrees and light winds from the west."
#     afconvert -f WAVE -d LEF32@48000 -c 2 weather.wav weather48k.wav
#     T="Ferrum is a runtime for large language models on Apple Silicon. It loads GGUF files, plans memory so the context fits beside the weights, and runs its own Metal kernels. Speculative decoding drafts several tokens at once and verifies them in one pass. Sessions keep the cache between turns so a long conversation does not pay for its history again."
#     say -o long.wav --data-format=LEI16@16000 "$T $T $T $T $T"
#     say -o toolong.wav --data-format=LEI16@16000 "$T $T $T $T $T $T $T $T $T $T $T"
#     afconvert -f WAVE -d LEF32@48000 -c 2 long.wav long48k.wav (and toolong.wav to toolong48k.wav)
#   S/regress/picture.png: docs/measurements/parallel-slots-2026-10/concurrency-tiel.png
set -u
S=${S:-/private/tmp/claude-501/-Users-austinwells-Documents-ferrum/63c705ed-5e6d-4036-805c-1f0616b0bc55/scratchpad}
L=$S/llama.cpp/build/bin
M=$HOME/models/hub/models--LiquidAI--LFM3-Audio-2B-GGUF/snapshots/main
LANG_FILE=$M/LFM3-Audio-2B-Q4_K_M.gguf
CHECK=target/release/examples/hybrid_audio_check
CLI=target/release/ferrum-cli
mkdir -p $S/cmp

echo "# start $(date -u '+%Y-%m-%d %H:%M:%S UTC'); commit $(git rev-parse --short HEAD); llama.cpp $(git -C $S/llama.cpp rev-parse --short HEAD)"
sysctl -n machdep.cpu.brand_string hw.memsize
echo "## encoder rows vs llama.cpp mtmd_get_output_embd (same 16 kHz samples)"
for q in BF16 Q8_0; do
  for c in fox weather weather48k long; do
    [[ $q == BF16 && $c == long ]] && continue
    P=$M/mmproj-LFM3-Audio-2B-$q.gguf
    echo "== $c $q"
    $CHECK $P $S/audio/$c.wav --samples $S/cmp/$c.f32 > /dev/null
    $S/embdump/embdump $LANG_FILE $P $S/cmp/$c.f32 $S/cmp/$c-$q.ref > /dev/null 2>&1
    $CHECK $P $S/audio/$c.wav $S/cmp/$c-$q.ref
  done
done
echo "== toolong Q8_0 (must be refused)"
$CHECK $M/mmproj-LFM3-Audio-2B-Q8_0.gguf $S/audio/toolong.wav 2>&1 | tail -1

echo "## memory plan against Metal's allocations, and the CPU front end on 48 kHz stereo float"
for q in BF16 Q8_0; do
  echo "== long48k $q --memory"
  $CHECK $M/mmproj-LFM3-Audio-2B-$q.gguf $S/audio/long48k.wav --memory
done
echo "== toolong48k Q8_0: refused from the header, wall time including the projector load"
/usr/bin/time -p $CHECK $M/mmproj-LFM3-Audio-2B-Q8_0.gguf $S/audio/toolong48k.wav 2>&1 | grep -E 'Error|real'

P=$M/mmproj-LFM3-Audio-2B-Q8_0.gguf
lm() { $L/llama-mtmd-cli -m $LANG_FILE --mmproj $P --jinja --temp 0 -n 96 "$@" 2>/dev/null | sed '/^$/d' | tail -2; }
fc() { $CLI --model $LANG_FILE --mmproj $P --temperature 0 --max-tokens 96 "$@" 2>&1 | grep -v -E '^(ferrum|model|projector|context|>|\s*$)' | tail -4; }
echo "## greedy, Q4_K_M + Q8_0 projector: llama-mtmd-cli --jinja vs ferrum-cli"
echo "== ASR fox"
lm --audio $S/audio/fox.wav -sys "Perform ASR in English." -p "<__media__>"
printf '/audio %s\n\n/exit\n' $S/audio/fox.wav | fc --system "Perform ASR in English."
echo "== ASR weather"
lm --audio $S/audio/weather.wav -sys "Perform ASR in English." -p "<__media__>"
printf '/audio %s\n\n/exit\n' $S/audio/weather.wav | fc --system "Perform ASR in English."
Q="What city is mentioned, and what is the expected high temperature? Answer in one sentence."
echo "== question"
lm --audio $S/audio/weather.wav -p "<__media__>$Q"
printf '/audio %s\n%s\n/exit\n' $S/audio/weather.wav "$Q" | fc
T="Two clips are attached. Summarise each in a few words."
echo "== two clips"
lm --audio $S/audio/fox.wav --audio $S/audio/weather48k.wav -p "<__media__><__media__>$T"
printf '/audio %s\n/audio %s\n%s\n/exit\n' $S/audio/fox.wav $S/audio/weather48k.wav "$T" | fc

# expand_media was split into splice_media and locate_media after the regression run:
# the same LFM3-VL picture check as regression-2026-10-30.sh, against its answer md5
# (5a55a681a5c85ce8813e746bd69616c5 on both builds there).
VL=$HOME/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main
echo "## LFM3-VL picture, greedy, as in the regression record"
printf '/image %s\nDescribe this picture.\n/exit\n' $S/regress/picture.png | \
  $CLI --model $VL/LFM3-VL-2B-Q4_K_M.gguf --mmproj $VL/mmproj-LFM3-VL-2B-Q8_0.gguf \
    --temperature 0 --max-tokens 96 2> /dev/null | grep -v -E 'tok/s|[0-9]+(\.[0-9]+)? ?(ms|s)\b' > $S/cmp/vl.clean
echo "answer md5 $(md5 -q $S/cmp/vl.clean)"
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
