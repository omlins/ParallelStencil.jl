module Diffusion3D_CUDAExt
    using Diffusion3D
    import CUDA
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(CUDA, Float64, 3)
    Diffusion3D.diffusion3D(backend::Type{<:CUDABackend}) = (@info "using CUDA backend"; diffusion3D())
    include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
end
