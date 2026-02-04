module ParallelStencil_KernelAbstractionsExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "KernelAbstractionsExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "KernelAbstractionsExt", "allocators.jl"))
end
