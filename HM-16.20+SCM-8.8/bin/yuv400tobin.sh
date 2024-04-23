#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./yuv400tobin.sh [input dir] [output dir1] [output dir2]"
fi

indir=$1
outdir1=$2  #bin
outdir2=$3  #yuv

mkdir outdir1
if [[ ! -d "${outdir1}" ]]; then
  echo "${outdir1} doesn't exist. Creating it.";
  mkdir -p ${outdir1}
fi
mkdir outdir2
if [[ ! -d "${outdir2}" ]]; then
  echo "${outdir2} doesn't exist. Creating it.";
  mkdir -p ${outdir2}
fi

for invideo in $(ls ${indir})
do
	inname="${indir}/${invideo}"
	Outname1="${outdir1}/${invideo:0:-4}"  #remove .yuv
    Outname2="${outdir2}/${invideo:0:-4}"  #remove .yuv
	outname_bin="${Outname1%.*}.bin"
    outname_yuv="${Outname2%.*}.yuv"
	echo "${inname}"
    echo "${outname_yuv}"
    echo "${outname_bin}"
    ./TAppEncoderStatic -i ${inname} -c ../cfg/encoder_lowdelay_main_rext_yuv400.cfg -c ../cfg/per-sequence/Ayuv400.cfg -b ${outname_bin} -o ${outname_yuv}

done