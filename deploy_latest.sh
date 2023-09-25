#!/usr/bin/env bash

current_version=$(python setup.py -V)

REPO=${REPO:-ammod_blob_detector}

echo "Uploading to ${REPO} ..."

twine upload \
	--repository ${REPO} \
	dist/*${current_version}.tar.gz \

ret_code=$?
if [[ $ret_code == 0 ]]; then
	echo "OK"
fi
