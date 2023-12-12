module CPU
    using ..Diffusion3D
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)
    Diffusion3D.diffusion3D(backend::Type{<:CPUBackend}) = (@info "using CPU backend"; diffusion3D())
    include(joinpath(@__DIR__, "..", "src", "backends", "diffusion3D.jl"))
end
