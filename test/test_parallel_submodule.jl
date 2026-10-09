using Test
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Threads, Float64, 2)

module KernelModule
    using ParallelStencil
    @init_parallel_stencil(Threads, Float64, 2)

    @doc "Copy each source element to the corresponding destination element." @parallel_indices (I...) function copy_kernel!(dst, src)
        @inbounds dst[I...] = src[I...]
        return nothing
    end
end

@testset "parallel kernel docstring in a submodule" begin
    @test only((@doc KernelModule.copy_kernel!).text) == "Copy each source element to the corresponding destination element."
end
