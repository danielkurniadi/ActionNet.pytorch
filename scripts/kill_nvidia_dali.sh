#! /bin/bash

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
