module Diffusion3D_KernelAbstractions
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(KernelAbstractions, Float64, 3)
    include(joinpath(@__DIR__, "diffusion3D.jl"))
end
