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

@testset "parallel kernel defined in a submodule" begin
    @test Base.Docs.hasdoc(KernelModule, :copy_kernel!)
    A = zeros(4, 4)
    B = ones(4, 4)
    @parallel (1:4, 1:4) KernelModule.copy_kernel!(A, B)
    @test A == B
end
