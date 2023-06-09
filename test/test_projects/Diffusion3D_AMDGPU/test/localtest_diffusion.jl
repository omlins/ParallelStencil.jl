push!(LOAD_PATH, "@stdlib")  # NOTE: this is needed to enable this test to run from the Pkg manager
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, "test_projects", "Diffusion3D_AMDGPU"))
Pkg.instantiate()
using Diffusion3D_AMDGPU
@test Diffusion3D_AMDGPU.diffusion3D()