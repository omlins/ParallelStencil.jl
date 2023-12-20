module Diffusion_AMDGPUExt
    module Diffusion2D
        using Diffusion
        import AMDGPU
        using ParallelStencil
        using ParallelStencil.FiniteDifferences2D
        @init_parallel_stencil(AMDGPU, Float64, 2)
        Diffusion.diffusion2D(backend::Type{<:AMDGPUBackend}) = (@info "using AMDGPU backend"; diffusion2D())
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion2D.jl"))
    end

    module Diffusion3D
        using Diffusion
        import AMDGPU
        using ParallelStencil
        using ParallelStencil.FiniteDifferences3D
        @init_parallel_stencil(package=AMDGPU, ndims=3)
        Diffusion.diffusion3D(backend::Type{<:AMDGPUBackend}, NumberType) = (@info "using AMDGPU backend"; diffusion3D(NumberType))
        include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
    end

    module memcopyXD
        using Diffusion
        import AMDGPU
        using ParallelStencil
        @init_parallel_stencil(package=AMDGPU)
        Diffusion.memcopy!(B::Data.Array, A::Data.Array) = (@info "using AMDGPU backend"; @parallel memcopy!(B, A))
        include(joinpath(@__DIR__, "..", "src", "backends", "memcopyXD.jl"))
    end
end
