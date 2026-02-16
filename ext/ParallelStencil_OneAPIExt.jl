module ParallelStencil_OneAPIExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "OneAPIExt", "select_hardware.jl"))
end
