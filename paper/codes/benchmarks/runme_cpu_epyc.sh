#!/bin/bash

# INFO: `-t auto` sets 32 physical cores on AMD EPYC CPUs
for ie in {1..20}; do
    julia8 --project -t auto --check-bounds=no -O3 diffusion3D_clean_benchm_explicit_cpu.jl
done

for ie in {1..20}; do
    julia8 --project -t auto --check-bounds=no -O3 diffusion3D_clean_benchm_cpu.jl
done

for ie in {1..20}; do
    julia8 --project -t auto --check-bounds=no -O3 diffusion3D_clean_benchm_cpuarrays.jl
done
