module Diffusion3D_Threads
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)
    include(joinpath(@__DIR__, "..", "..", "shared", "diffusion3D.jl"))
    ParallelStencil.@reset_parallel_stencil()
end
