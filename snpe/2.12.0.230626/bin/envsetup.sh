#!/bin/bash
#==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script sets up the various environment variables needed to run sdk binaries and scripts
OPTIND=1

_usage()
{
cat << EOF
Usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h]

Script sets up environment variables needed for running sdk binaries and scripts

EOF
}

function _setup_aisw_sdk()
{
  # get directory of the bash script
  local SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  local AISW_SDK_ROOT=$(readlink -f ${SOURCEDIR}/..)
  export SNPE_ROOT="$( cd "${AISW_SDK_ROOT}" && pwd )"
  export PYTHONPATH="${AISW_SDK_ROOT}/lib/python/":${PYTHONPATH}
  export PATH=${AISW_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}
  if [ "x${LD_LIBRARY_PATH}" = "x" ]; then
    export LD_LIBRARY_PATH=${AISW_SDK_ROOT}/lib/x86_64-linux-clang
  else
    export LD_LIBRARY_PATH=${AISW_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
  fi
}

function _cleanup()
{
  unset -f _usage
  unset -f _setup_aisw_sdk
  unset -f _cleanup
}

# parse arguments
while getopts "h?" opt; do
  case ${opt} in
    h  ) _usage; return 0 ;;
    \? ) echo "See -h for help."; return 1 ;;
  esac
done

_setup_aisw_sdk

# cleanup
_cleanup

echo "[INFO] AISW SDK environment set"
echo "[INFO] SNPE_ROOT: ${SNPE_ROOT}"

