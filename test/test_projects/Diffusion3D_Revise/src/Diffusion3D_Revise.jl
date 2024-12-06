module Diffusion3D_Revise
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)
    include(joinpath(@__DIR__, "diffusion3D_tmp.jl"))
end
