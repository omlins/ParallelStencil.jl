module Diffusion_CUDAExt
    module Diffusion2D
        using Diffusion
        import CUDA
        using ParallelStencil
        using ParallelStencil.FiniteDifferences2D
        @init_parallel_stencil(CUDA, Float64, 2)
        Diffusion.diffusion2D(backend::Type{<:CUDABackend}) = (@info "using CUDA backend"; diffusion2D())
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion2D.jl"))
    end

    module Diffusion3D
        using Diffusion
        import CUDA
        using ParallelStencil
        using ParallelStencil.FiniteDifferences3D
        @init_parallel_stencil(package=CUDA, ndims=3)
        Diffusion.diffusion3D(backend::Type{<:CUDABackend}, NumberType) = (@info "using CUDA backend"; diffusion3D(NumberType))
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
    end

    module memcopyXD
        using Diffusion
        import CUDA
        using ParallelStencil
        @init_parallel_stencil(package=CUDA)
        Diffusion.memcopy!(B::Data.Array, A::Data.Array) = (@info "using CUDA backend"; @parallel memcopy!(B, A))
        include(joinpath(@__DIR__, "..", "src", "backends", "memcopyXD.jl"))
    end
end
