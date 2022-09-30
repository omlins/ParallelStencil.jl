#!/bin/bash

# WARNING: nx=ny=nz=1024 for best perf on A100 40GB
for ie in {1..20}; do
    julia8 --project --check-bounds=no -O3 diffusion3D_clean_benchm_explicit.jl
done

# WARNING: nx=ny=nz=1024 for best perf on A100 40GB
for ie in {1..20}; do
    julia8 --project --check-bounds=no -O3 diffusion3D_clean_benchm.jl
done

for ie in {1..20}; do
    julia8 --project --check-bounds=no -O3 diffusion3D_clean_benchm_gpuarrays.jl
done
