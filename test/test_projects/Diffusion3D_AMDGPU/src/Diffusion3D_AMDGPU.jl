module Diffusion3D_AMDGPU
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(AMDGPU, Float64, 3)
    include(joinpath(@__DIR__, "..", "..", "shared", "diffusion3D.jl"))    
end
