#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

for directory in "apps" "kaminpar" "dkaminpar" "tests" "dtests" "library"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec clang-format -i {} \;
done