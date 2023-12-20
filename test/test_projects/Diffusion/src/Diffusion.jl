module Diffusion
    export CPUBackend, CUDABackend, AMDGPUBackend, diffusion2D, diffusion3D, memcopy!

    abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct CUDABackend <: AbstractBackend end
    struct AMDGPUBackend <: AbstractBackend end

    diffusion2D(backend::Type{<:AbstractBackend}) = error("module not loaded or diffusion2D not implemented for backend: $backend")
    diffusion3D(backend::Type{<:AbstractBackend}, NumberType) = error("module not loaded or diffusion3D not implemented for backend: $backend")
    memcopy!(B, A) = error("module not loaded or memcopy! not implemented for types: $(typeof(A)) and $(typeof(B))") 

    include("CPU.jl")
end
