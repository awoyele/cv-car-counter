#!/usr/bin/env bash
# Download the sample Pexels clip used by configs/pexels_13361265.yaml.
#
# Source: https://www.pexels.com/video/cars-on-alley-in-city-13361265/
# Saved as: videos/13361265-hd_1920_1080_60fps.mp4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${ROOT}/videos"
DEST="${DEST_DIR}/13361265-hd_1920_1080_60fps.mp4"
PAGE_URL="https://www.pexels.com/video/cars-on-alley-in-city-13361265/"
DOWNLOAD_URL="https://www.pexels.com/download/video/13361265/"
USER_AGENT="Mozilla/5.0 (compatible; cv-car-counter/0.1; +${PAGE_URL})"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [--force]

Download the sample Pexels traffic clip into videos/.

  --force   Re-download even if the file already exists.
EOF
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  echo "Usage: $(basename "$0") [--force]" >&2
  exit 2
fi

if [[ -f "${DEST}" && "${FORCE}" -eq 0 ]]; then
  echo "Already present: ${DEST}"
  echo "Re-run with --force to download again."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download the sample video." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
TMP="${DEST}.part"
trap 'rm -f "${TMP}"' EXIT

echo "Downloading sample clip from ${PAGE_URL}"
curl -fL --retry 3 --retry-delay 2 \
  -A "${USER_AGENT}" \
  -o "${TMP}" \
  "${DOWNLOAD_URL}"

if [[ ! -s "${TMP}" ]]; then
  echo "Download failed: empty file." >&2
  exit 1
fi

mv "${TMP}" "${DEST}"
trap - EXIT
echo "Saved ${DEST}"
echo "License: free to use under the Pexels License (${PAGE_URL})"
