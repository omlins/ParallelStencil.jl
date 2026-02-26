module ParallelStencil_AMDGPUExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "AMDGPUExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "AMDGPUExt", "select_hardware.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "AMDGPUExt", "allocators.jl"))
end