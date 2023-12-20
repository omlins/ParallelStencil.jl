module CPU
    module Diffusion2D
        using Diffusion
        using ParallelStencil
        using ParallelStencil.FiniteDifferences2D
        @init_parallel_stencil(Threads, Float64, 2)
        Diffusion.diffusion2D(backend::Type{<:CPUBackend}) = (@info "using CPU backend"; diffusion2D())
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion2D.jl"))
    end

    module Diffusion3D
        using Diffusion
        using ParallelStencil
        using ParallelStencil.FiniteDifferences3D
        @init_parallel_stencil(package=Threads, ndims=3)
        Diffusion.diffusion3D(backend::Type{<:CPUBackend}, NumberType) = (@info "using CPU backend"; diffusion3D(NumberType))
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
    end

    module memcopyXD
        using Diffusion
        using ParallelStencil
        @init_parallel_stencil(package=Threads)
        Diffusion.memcopy!(B::Data.Array, A::Data.Array) = (@info "using CPU backend"; @parallel memcopy!(B, A))
        include(joinpath(@__DIR__, "..", "src", "backends", "memcopyXD.jl"))
    end
end
