module Diffusion3D
    export CPUBackend, CUDABackend, AMDGPUBackend, diffusion3D

    abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct CUDABackend <: AbstractBackend end
    struct AMDGPUBackend <: AbstractBackend end

    diffusion3D(backend::Type{<:AbstractBackend}) = error("module not loaded or diffusion3D not implemented for backend: $backend")
    include("CPU.jl")
end
