push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using Diffusion
@test diffusion2D(CPUBackend) <: Array
@test diffusion3D(CPUBackend, Float32) <: Array
B = zeroes(4, 4)
A = ones(4, 4)
memcopy!(B, A)
@test all(B .== A)