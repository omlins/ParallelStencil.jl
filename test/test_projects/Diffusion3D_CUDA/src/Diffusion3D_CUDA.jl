module Diffusion3D_CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(CUDA, Float64, 3)
    include(joinpath(@__DIR__, "..", "..", "shared", "diffusion3D.jl"))
end
