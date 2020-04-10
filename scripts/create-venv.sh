#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

python_name="$(basename "${src_dir}" | sed -e 's/-//' | sed -e 's/-/_/g')"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"
download="${src_dir}/download"

cpu_arch="$(uname -m)"
target_arch="$("${src_dir}/architecture.sh")"

# -----------------------------------------------------------------------------

# Create virtual environment
echo "Creating virtual environment at ${venv}"
rm -rf "${venv}"
python3 -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies"
pip3 ${PIP_INSTALL} --upgrade pip
pip3 ${PIP_INSTALL} --upgrade wheel setuptools

# Install local Rhasspy dependencies if available
grep '^rhasspy-' "${src_dir}/requirements.txt" | \
    xargs pip3 ${PIP_INSTALL} -f "${download}"

pip3 ${PIP_INSTALL} -r requirements.txt

# Kenlm
if [[ -n "$(command -v build_binary)" ]]; then
    kenlm_file="${download}/kenlm-20200308_${target_arch}.tar.gz"
    if [[ ! -s "${kenlm_file}" ]]; then
        echo "Downloading KenLM (${kenlm_file})"
        curl -sSfL -o "${kenlm_file}" "https://github.com/synesthesiam/docker-kenlm/releases/download/v2020.03.28/kenlm-20200308_${target_arch}.tar.gz"
    fi

    tar -C "${venv}/bin" -xf "${kenlm_file}" build_binary
fi

# Mozilla Native Client
if [[ -n "$(command -v generate_trie)" ]]; then
    native_client_file="${download}/native_client.${target_arch}.cpu.linux.tar.xz"
    if [[ ! -s "${native_client_file}" ]]; then
        echo "Downloading DeepSpeech native client (${native_client_file})"
        curl -sSfL -o "${native_client_file}" "https://github.com/mozilla/DeepSpeech/releases/download/v0.6.0/native_client.${target_arch}.cpu.linux.tar.xz"
    fi

    tar -C "${venv}/bin" -xf "${native_client_file}" generate_trie
fi

# Optional development requirements
pip3 ${PIP_INSTALL} -r requirements_dev.txt || \
    echo "Failed to install development requirements"

# -----------------------------------------------------------------------------

echo "OK"
