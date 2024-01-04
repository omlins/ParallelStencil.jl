push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
import AMDGPU
using Diffusion
@test diffusion2D(AMDGPUBackend) <: AMDGPU.ROCArray
@test diffusion3D(AMDGPUBackend, Float32) <: AMDGPU.ROCArray
B = AMDGPU.zeros(4, 4)
A = AMDGPU.ones(4, 4)
memcopy!(B, A)
@test all(B .== A)
D = AMDGPU.zeros(4, 4, 4)
C = AMDGPU.ones(4, 4, 4)
memcopy!(D, C)
@test all(D .== C)
