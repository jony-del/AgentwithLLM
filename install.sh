#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.11.28"
REPOSITORY="https://github.com/jony-del/AgentwithLLM"
VERSION="latest"
TEMP_ROOT=""
FORWARD_ARGS=()

while (($#)); do
  case "$1" in
    --version)
      [[ $# -ge 2 ]] || { echo "--version requires a tag" >&2; exit 2; }
      VERSION="$2"
      shift 2
      ;;
    --dev|--upgrade|--check|--dry-run|--skip-sandbox|--non-interactive)
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cleanup() {
  if [[ -n "$TEMP_ROOT" && -d "$TEMP_ROOT" ]]; then
    rm -rf -- "$TEMP_ROOT"
  fi
}
trap cleanup EXIT

download() {
  local url="$1" destination="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$destination" || { echo "download failed: $url" >&2; exit 10; }
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$destination" "$url" || { echo "download failed: $url" >&2; exit 10; }
  else
    echo "curl or wget is required to download the installer" >&2
    exit 10
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/installer/install.py" ]]; then
  SOURCE_ROOT="$SCRIPT_DIR"
else
  if printf '%s\n' "${FORWARD_ARGS[@]}" | grep -qx -- '--dev'; then
    echo "--dev requires a persistent source checkout" >&2
    exit 2
  fi
  TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/polaris-install.XXXXXX")"
  if [[ "$VERSION" == "latest" ]]; then
    BASE="$REPOSITORY/releases/latest/download"
  else
    BASE="$REPOSITORY/releases/download/$VERSION"
  fi
  ARCHIVE="$TEMP_ROOT/polaris-source.tar.gz"
  SUMS="$TEMP_ROOT/SHA256SUMS"
  echo "Downloading Polaris $VERSION release..."
  download "$BASE/polaris-source.tar.gz" "$ARCHIVE"
  download "$BASE/SHA256SUMS" "$SUMS"
  EXPECTED="$(awk '{name=$2; sub(/^\*/, "", name); if (name=="polaris-source.tar.gz") {print $1; exit}}' "$SUMS")"
  [[ -n "$EXPECTED" ]] || { echo "SHA256SUMS does not contain polaris-source.tar.gz" >&2; exit 10; }
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  else
    ACTUAL="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
  fi
  [[ "$ACTUAL" == "$EXPECTED" ]] || { echo "SHA-256 mismatch for polaris-source.tar.gz" >&2; exit 10; }
  SOURCE_ROOT="$TEMP_ROOT/source"
  mkdir -p "$SOURCE_ROOT"
  tar -xzf "$ARCHIVE" -C "$SOURCE_ROOT"
  [[ -f "$SOURCE_ROOT/installer/install.py" ]] || { echo "release archive is missing installer/install.py" >&2; exit 10; }
fi

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  if printf '%s\n' "${FORWARD_ARGS[@]}" | grep -Eqx -- '--check|--dry-run'; then
    echo "uv is missing; check/dry-run mode will not install it" >&2
    exit 10
  fi
  echo "Installing uv $UV_VERSION..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh || exit 10
  else
    wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | sh || exit 10
  fi
  UV="${HOME}/.local/bin/uv"
  [[ -x "$UV" ]] || UV="$(command -v uv || true)"
  [[ -n "$UV" ]] || { echo "uv installation completed but uv was not found" >&2; exit 10; }
fi

if ! printf '%s\n' "${FORWARD_ARGS[@]}" | grep -Eqx -- '--check|--dry-run'; then
  "$UV" python install 3.12 || exit 10
else
  export UV_PYTHON_DOWNLOADS=never
fi
PYTHON="$("$UV" python find 3.12 | tail -n 1)"
[[ -x "$PYTHON" ]] || { echo "uv did not return a Python 3.12 executable" >&2; exit 10; }
"$PYTHON" "$SOURCE_ROOT/installer/install.py" --source "$SOURCE_ROOT" "${FORWARD_ARGS[@]}"
