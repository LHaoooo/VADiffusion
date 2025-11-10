#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./avitoavi.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir outdir
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

for invideo in $(ls ${indir})
do
	inname="${indir}/${invideo}"
	outname="${outdir}/${invideo}"  
	echo "${inname}"
    echo "${outname}"
	ffmpeg -i ${inname} -s 256x256 ${outname} 
done