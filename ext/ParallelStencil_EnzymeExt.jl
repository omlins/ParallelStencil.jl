module ParallelStencil_EnzymeExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "EnzymeExt", "autodiff_gpu.jl"))
end