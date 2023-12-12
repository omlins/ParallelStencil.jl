module AMDGPUExt
    using Diffusion3D
    import AMDGPU
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(AMDGPU, Float64, 3)
    Diffusion3D.diffusion3D(backend::Type{<:AMDGPUBackend}) = (@info "using AMDGPU backend"; diffusion3D())
    include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
end
