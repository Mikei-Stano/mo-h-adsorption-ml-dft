#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this workflow on a fresh Linux node.
# Creates a pyenv-based environment and installs all required Python deps.
#
# Usage examples:
#   bash scripts/setup_pyenv_env.sh
#   ENV_NAME=cemea-env PYTHON_VERSION=3.10.16 bash scripts/setup_pyenv_env.sh
#   INSTALL_SYSTEM_DEPS=1 bash scripts/setup_pyenv_env.sh

ENV_NAME="${ENV_NAME:-cemea-env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.16}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

log() {
  echo "[setup] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[setup] Missing command: $1" >&2
    exit 1
  fi
}

install_system_deps_debian() {
  log "Installing system packages (Debian/Ubuntu)..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libxc-dev \
    libscalapack-mpi-dev \
    libopenmpi-dev \
    openmpi-bin \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev
}

maybe_install_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    log "Skipping system package installation (set INSTALL_SYSTEM_DEPS=1 to enable)."
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    log "sudo not available; cannot auto-install system dependencies."
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    install_system_deps_debian
  else
    log "Unsupported package manager for auto-install. Install compiler/BLAS/MPI deps manually."
  fi
}

install_pyenv_if_missing() {
  if command -v pyenv >/dev/null 2>&1; then
    log "pyenv already installed: $(pyenv --version)"
    return
  fi

  log "pyenv not found. Installing pyenv into ~/.pyenv ..."
  require_cmd curl
  curl -fsSL https://pyenv.run | bash

  export PYENV_ROOT="${HOME}/.pyenv"
  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)"

  log "pyenv installed."
}

init_pyenv_shell() {
  export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
  export PATH="${PYENV_ROOT}/bin:${PATH}"

  require_cmd pyenv
  eval "$(pyenv init -)"

  # Optional plugin init if available.
  if command -v pyenv-virtualenv-init >/dev/null 2>&1; then
    eval "$(pyenv virtualenv-init -)"
  fi
}

create_env() {
  log "Ensuring Python ${PYTHON_VERSION} exists in pyenv..."
  pyenv install -s "${PYTHON_VERSION}"

  if pyenv commands | grep -q '^virtualenv$'; then
    if ! pyenv versions --bare | grep -qx "${ENV_NAME}"; then
      log "Creating pyenv virtualenv: ${ENV_NAME}"
      pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
    else
      log "pyenv env ${ENV_NAME} already exists."
    fi
    PY_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
    PIP_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/pip"
  else
    # Fallback: create a venv under pyenv versions path.
    if [[ ! -x "${PYENV_ROOT}/versions/${ENV_NAME}/bin/python" ]]; then
      log "pyenv-virtualenv plugin not found; creating venv at ${PYENV_ROOT}/versions/${ENV_NAME}"
      "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python" -m venv "${PYENV_ROOT}/versions/${ENV_NAME}"
    fi
    PY_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/python"
    PIP_BIN="${PYENV_ROOT}/versions/${ENV_NAME}/bin/pip"
  fi

  log "Using Python: ${PY_BIN}"
  "${PY_BIN}" --version
}

install_python_deps() {
  log "Upgrading pip/setuptools/wheel/cython..."
  "${PIP_BIN}" install --upgrade pip setuptools wheel cython

  log "Installing workflow requirements from ${REQ_FILE}"
  "${PIP_BIN}" install -r "${REQ_FILE}"
}

verify_install() {
  log "Verifying imports..."
  "${PY_BIN}" - <<'PY'
import importlib
mods = ["ase", "numpy", "pandas", "gpaw", "pymatgen"]
for m in mods:
    importlib.import_module(m)
print("All required Python modules import successfully.")
PY

  log "Done. Activate with one of:"
  echo "  pyenv activate ${ENV_NAME}"
  echo "  source \"${PYENV_ROOT}/versions/${ENV_NAME}/bin/activate\""
}

main() {
  log "Project root: ${PROJECT_ROOT}"
  maybe_install_system_deps
  install_pyenv_if_missing
  init_pyenv_shell
  create_env
  install_python_deps
  verify_install
}

main "$@"
