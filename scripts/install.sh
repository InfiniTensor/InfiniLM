#!/bin/bash
set -e  # Exit on error

# Check if project path is provided
if [ -z "$1" ]; then
    echo "Usage: source deploy.sh PROJECT_PATH [XMAKE_CONFIG_FLAGS]"
    exit 1
fi

# Set INFINI_ROOT
export INFINI_ROOT="$HOME/.infini"

# Check if INFINI_ROOT/bin is already in PATH, if not, add it
case ":$PATH:" in
  *":$INFINI_ROOT/bin:"*) ;; # Already in PATH, do nothing
  *) export PATH="$INFINI_ROOT/bin:$PATH" ;; # Add to PATH
esac

# Check if INFINI_ROOT/lib is already in LD_LIBRARY_PATH, if not, add it
case ":$LD_LIBRARY_PATH:" in
  *":$INFINI_ROOT/lib:"*) ;; # Already in LD_LIBRARY_PATH, do nothing
  *) export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH" ;; # Add to LD_LIBRARY_PATH
esac

# Change to project directory
cd "$1"

# Shift first argument (project path) and pass the rest to xmake
shift
xmake clean -a
xmake f "$@" -cv
xmake
xmake install

xmake build infiniop-test
xmake install infiniop-test
