push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
import CUDA
using Diffusion
@test diffusion2D(CUDABackend) <: CUDA.CuArray
@test diffusion3D(CUDABackend, Float32) <: CUDA.CuArray
B = CUDA.zeros(4, 4)
A = CUDA.ones(4, 4)
memcopy!(B, A)
@test all(B .== A)
D = CUDA.zeros(4, 4, 4)
C = CUDA.ones(4, 4, 4)
memcopy!(D, C)
@test all(D .== C)
