#!/bin/bash

# Exit on any error.
set -e

# Resolve Project Root
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

DEFAULT_INSTALL_PATH="$HOME/.infini"
INSTALL_PREFIX="${INSTALL_PREFIX:-$DEFAULT_INSTALL_PATH}"
BUILD_DIR="${BUILD_DIR:-build}"

update_bashrc_export() {
    local name="$1"
    local value="$2"
    local escaped_value
    escaped_value=$(printf '%s' "$value" | sed 's/[\\&|]/\\&/g')

    if grep -q "^export ${name}=" "$HOME/.bashrc"; then
        sed -i "s|^export ${name}=.*|export ${name}=\"$escaped_value\"|" "$HOME/.bashrc"
    else
        echo "export ${name}=\"$value\"" >> "$HOME/.bashrc"
    fi
}

echo "========================================================"
echo " Starting InfiniCCL Build"
echo " Build Directory:  $BUILD_DIR"
echo " Install Prefix:   $INSTALL_PREFIX"
echo "========================================================"

# Create directories if they don't exist.
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_PREFIX"

# Configure and Build
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"

cmake --build "$BUILD_DIR" -j$(nproc)
cmake --install "$BUILD_DIR"

# Handle Environment Variables (`PATH` and `LD_LIBRARY_PATH`)
BIN_PATH="$INSTALL_PREFIX/bin"
LIB_PATH="$INSTALL_PREFIX/lib"

# Check if `PATH` already contains the `bin` path.
if [[ ":$PATH:" != *":$BIN_PATH:"* ]]; then
    echo "--> Adding $BIN_PATH to current session PATH..."
    export PATH="$BIN_PATH:$PATH"
fi

# Automatically update `~/.bashrc` for the user.
if ! grep -q "$BIN_PATH" "$HOME/.bashrc"; then
    echo "--> Updating ~/.bashrc with InfiniCCL paths..."
    echo "" >> "$HOME/.bashrc"
    echo "# InfiniCCL Paths" >> "$HOME/.bashrc"
    echo "export PATH=\"$BIN_PATH:\$PATH\"" >> "$HOME/.bashrc"
    echo "export LD_LIBRARY_PATH=\"$LIB_PATH:\$LD_LIBRARY_PATH\"" >> "$HOME/.bashrc"
    echo "Successfully updated ~/.bashrc. Please run 'source ~/.bashrc' or restart your terminal."
fi

echo "--> Updating INFINICCL_ROOT in ~/.bashrc..."
update_bashrc_export "INFINICCL_ROOT" "$PROJECT_ROOT"
update_bashrc_export "InfiniCCL_ROOT" "$PROJECT_ROOT"

echo "========================================================"
echo " Build and Install Finished!"
echo " Binaries:  $BIN_PATH"
echo " Libraries: $LIB_PATH"
echo " Headers:   $INSTALL_PREFIX/include"
echo "========================================================"
echo " NOTE: If this is your first install or there's a path update, run: source ~/.bashrc"
echo " You can now run 'icclrun' from anywhere."
