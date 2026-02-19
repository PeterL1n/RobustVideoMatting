#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

variant="mobilenetv3"
device="cuda"
input_source=""
checkpoint=""
output_dir="${SCRIPT_DIR}/test_outputs"
downsample_ratio=""
seq_chunk="12"
output_video_mbps="4"
input_resize=()

print_usage() {
  cat <<'EOF'
Usage:
  ./run_inference_test.sh --input <video_or_image_sequence>

Options:
  --input <path>              Input video file or image sequence directory. (required)
  --variant <name>            Model variant: mobilenetv3 | resnet50 (default: mobilenetv3)
  --device <name>             Torch device, e.g. cuda, cuda:0, cpu (default: cuda)
  --checkpoint <path>         Checkpoint path. If omitted, downloads to ./checkpoints/rvm_<variant>.pth
  --output-dir <path>         Output directory (default: ./test_outputs)
  --downsample-ratio <float>  Optional ratio in (0, 1]
  --seq-chunk <int>           Number of frames per inference chunk (default: 12)
  --output-video-mbps <int>   Output bitrate in Mbps (default: 4)
  --input-resize <w> <h>      Optional resize before inference
  -h, --help                  Show this help

Examples:
  ./run_inference_test.sh --input ./input.mp4
  ./run_inference_test.sh --input ./input.mp4 --device cpu --downsample-ratio 0.25
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || die "--input requires a value"
      input_source="$2"
      shift 2
      ;;
    --variant)
      [[ $# -ge 2 ]] || die "--variant requires a value"
      variant="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      device="$2"
      shift 2
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || die "--checkpoint requires a value"
      checkpoint="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --downsample-ratio)
      [[ $# -ge 2 ]] || die "--downsample-ratio requires a value"
      downsample_ratio="$2"
      shift 2
      ;;
    --seq-chunk)
      [[ $# -ge 2 ]] || die "--seq-chunk requires a value"
      seq_chunk="$2"
      shift 2
      ;;
    --output-video-mbps)
      [[ $# -ge 2 ]] || die "--output-video-mbps requires a value"
      output_video_mbps="$2"
      shift 2
      ;;
    --input-resize)
      [[ $# -ge 3 ]] || die "--input-resize requires <w> <h>"
      input_resize=("$2" "$3")
      shift 3
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${input_source}" ]] || die "--input is required"
[[ -e "${input_source}" ]] || die "input path does not exist: ${input_source}"
[[ "${variant}" == "mobilenetv3" || "${variant}" == "resnet50" ]] || die "--variant must be mobilenetv3 or resnet50"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "python binary not found: ${PYTHON_BIN}"
fi

if [[ -z "${checkpoint}" ]]; then
  checkpoint="${SCRIPT_DIR}/checkpoints/rvm_${variant}.pth"
fi

if [[ ! -f "${checkpoint}" ]]; then
  checkpoint_url="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_${variant}.pth"
  mkdir -p "$(dirname "${checkpoint}")"
  echo "checkpoint not found, downloading: ${checkpoint_url}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "${checkpoint_url}" -o "${checkpoint}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${checkpoint}" "${checkpoint_url}"
  else
    die "please install curl or wget, or pass --checkpoint <path>"
  fi
fi

mkdir -p "${output_dir}"

cmd=(
  "${PYTHON_BIN}" inference.py
  --variant "${variant}"
  --checkpoint "${checkpoint}"
  --device "${device}"
  --input-source "${input_source}"
  --output-type video
  --output-composition "${output_dir}/composition.mp4"
  --output-alpha "${output_dir}/alpha.mp4"
  --output-foreground "${output_dir}/foreground.mp4"
  --output-video-mbps "${output_video_mbps}"
  --seq-chunk "${seq_chunk}"
)

if [[ -n "${downsample_ratio}" ]]; then
  cmd+=(--downsample-ratio "${downsample_ratio}")
fi

if [[ ${#input_resize[@]} -eq 2 ]]; then
  cmd+=(--input-resize "${input_resize[0]}" "${input_resize[1]}")
fi

echo "running inference..."
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "done. outputs written to: ${output_dir}"
