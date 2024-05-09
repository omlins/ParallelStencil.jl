module Diffusion3D_Polyester
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Polyester, Float64, 3)
    include(joinpath(@__DIR__, "..", "..", "shared", "diffusion3D.jl"))
end
