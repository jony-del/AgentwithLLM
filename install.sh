#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.11.28"
REPOSITORY="https://github.com/jony-del/AgentwithLLM"
VERSION="latest"
TEMP_ROOT=""
FORWARD_ARGS=()
UNINSTALL=0
PURGE_DATA=0
ASSUME_YES=0
DEV=0
UPGRADE=0
CHECK=0
DRY_RUN=0
SKIP_SANDBOX=0
NON_INTERACTIVE=0

while (($#)); do
  case "$1" in
    --version)
      [[ $# -ge 2 ]] || { echo "--version requires a tag" >&2; exit 2; }
      VERSION="$2"
      shift 2
      ;;
    --dev)
      DEV=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --upgrade)
      UPGRADE=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --check)
      CHECK=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --skip-sandbox)
      SKIP_SANDBOX=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --non-interactive)
      NON_INTERACTIVE=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    --purge-data)
      PURGE_DATA=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ((UNINSTALL)); then
  if ((DEV || UPGRADE || CHECK || SKIP_SANDBOX)); then
    echo "[usage] --uninstall cannot be combined with --dev, --upgrade, --check, or --skip-sandbox" >&2
    exit 2
  fi
  if ((NON_INTERACTIVE && !ASSUME_YES && !DRY_RUN)); then
    echo "[usage] non-interactive uninstall requires --yes (or use --dry-run)" >&2
    exit 2
  fi
elif ((PURGE_DATA || ASSUME_YES)); then
  echo "[usage] --purge-data and --yes require --uninstall" >&2
  exit 2
fi

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
  if ((UNINSTALL)) && [[ ! -f "$SOURCE_ROOT/agent_core/uninstall.py" ]]; then
    echo "source checkout is missing agent_core/uninstall.py" >&2
    exit 10
  fi
else
  if ((DEV)); then
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
  [[ -f "$SOURCE_ROOT/agent_core/uninstall.py" ]] || { echo "release archive is missing agent_core/uninstall.py" >&2; exit 10; }
fi

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  if ((UNINSTALL || CHECK || DRY_RUN)); then
    echo "uv is missing; uninstall/check/dry-run mode will not install it" >&2
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

if ((!UNINSTALL && !CHECK && !DRY_RUN)); then
  "$UV" python install 3.12 || exit 10
else
  export UV_PYTHON_DOWNLOADS=never
fi
PYTHON="$("$UV" python find 3.12 | tail -n 1)"
[[ -x "$PYTHON" ]] || { echo "uv did not return a Python 3.12 executable" >&2; exit 10; }
if ((UNINSTALL)); then
  UNINSTALL_ARGS=()
  if ((PURGE_DATA)); then
    UNINSTALL_ARGS+=(--purge-data)
  fi
  if ((ASSUME_YES)); then
    UNINSTALL_ARGS+=(--yes)
  fi
  if ((DRY_RUN)); then
    UNINSTALL_ARGS+=(--dry-run)
  fi
  if ((NON_INTERACTIVE)); then
    UNINSTALL_ARGS+=(--non-interactive)
  fi
  "$PYTHON" "$SOURCE_ROOT/agent_core/uninstall.py" "${UNINSTALL_ARGS[@]}"
else
  "$PYTHON" "$SOURCE_ROOT/installer/install.py" --source "$SOURCE_ROOT" "${FORWARD_ARGS[@]}"
fi
