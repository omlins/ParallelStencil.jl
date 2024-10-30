module ParallelStencil_MetalExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "MetalExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "MetalExt", "allocators.jl"))
end