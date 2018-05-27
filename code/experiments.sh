# !/bin/bash

LAYERS=("6")
ACTIVATIONS=("tanh" "relu")
INITIALIZERS=("xavier")
NORMALIZERS=("batch_normalization")

# http://stackoverflow.com/questions/2870992/automatic-exit-from-bash-shell-script-on-error
set -e
for L in "${LAYERS[@]}"
do
    for A in "${ACTIVATIONS[@]}"
    do
        for I in "${INITIALIZERS[@]}"
        do
            for N in "${NORMALIZERS[@]}"
            do
                python3 benchmark.py -layers=$L -activation=$A -initializer=$I -normalizer=$N
            done
        done
    done
done
