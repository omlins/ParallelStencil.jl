push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using Diffusion3D_CUDA
@test Diffusion3D_CUDA.diffusion3D()