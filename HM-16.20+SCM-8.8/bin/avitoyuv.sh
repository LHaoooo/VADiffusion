#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./avitoyuv.sh [input dir] [output dir]"
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
	Outname="${outdir}/${invideo:0:-4}"  #remove .avi
	outname="${Outname%.*}.yuv"
	echo "${inname}"
    echo "${outname}"
	ffmpeg -i ${inname} -pix_fmt yuv420p -s 856*480 -y ${outname} 
done