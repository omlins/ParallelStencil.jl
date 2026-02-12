using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS, @select_hardware, @current_hardware
import ParallelStencil: @require, @prettystring, @iscpu
import ParallelStencil.ParallelKernel: handle

TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_CUDA, TEST_PACKAGES)
    end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional()
        TEST_PACKAGES = filter!(x -> x ≠ PKG_AMDGPU, TEST_PACKAGES)
    end
end
@static if PKG_METAL in TEST_PACKAGES
    import Metal
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU())
        TEST_PACKAGES = filter!(x -> x ≠ PKG_KERNELABSTRACTIONS, TEST_PACKAGES)
    end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions()

@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64

    eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @require !@is_initialized()
        @init_parallel_stencil($package, $FloatDefault, 3)
        @require @is_initialized()

        @testset "Pass-through macro mapping" begin
            @test @prettystring(1, @gridDim()) == "ParallelStencil.ParallelKernel.@gridDim"
            @test @prettystring(1, @blockIdx()) == "ParallelStencil.ParallelKernel.@blockIdx"
            @test @prettystring(1, @blockDim()) == "ParallelStencil.ParallelKernel.@blockDim"
            @test @prettystring(1, @threadIdx()) == "ParallelStencil.ParallelKernel.@threadIdx"
            @test @prettystring(1, @sync_threads()) == "ParallelStencil.ParallelKernel.@sync_threads"
            @test @prettystring(1, @sharedMem(T, dims)) == "ParallelStencil.ParallelKernel.@sharedMem T dims"
            @test @prettystring(1, @ps_show args) == "ParallelStencil.ParallelKernel.@pk_show args"
            @test @prettystring(1, @ps_println args) == "ParallelStencil.ParallelKernel.@pk_println args"
            @test @prettystring(1, @∀ i ∈ (x, y) body) == "ParallelStencil.ParallelKernel.@∀ i ∈ (x, y) body"

            @test @prettystring(1, @warpsize()) == "ParallelStencil.ParallelKernel.@warpsize"
            @test @prettystring(1, @laneid()) == "ParallelStencil.ParallelKernel.@laneid"
            @test @prettystring(1, @active_mask()) == "ParallelStencil.ParallelKernel.@active_mask"
            @test @prettystring(1, @shfl_sync(mask, val, lane)) == "ParallelStencil.ParallelKernel.@shfl_sync mask val lane"
            @test @prettystring(1, @shfl_sync(mask, val, lane, width)) == "ParallelStencil.ParallelKernel.@shfl_sync mask val lane width"
            @test @prettystring(1, @shfl_up_sync(mask, val, delta)) == "ParallelStencil.ParallelKernel.@shfl_up_sync mask val delta"
            @test @prettystring(1, @shfl_up_sync(mask, val, delta, width)) == "ParallelStencil.ParallelKernel.@shfl_up_sync mask val delta width"
            @test @prettystring(1, @shfl_down_sync(mask, val, delta)) == "ParallelStencil.ParallelKernel.@shfl_down_sync mask val delta"
            @test @prettystring(1, @shfl_down_sync(mask, val, delta, width)) == "ParallelStencil.ParallelKernel.@shfl_down_sync mask val delta width"
            @test @prettystring(1, @shfl_xor_sync(mask, val, lanemask)) == "ParallelStencil.ParallelKernel.@shfl_xor_sync mask val lanemask"
            @test @prettystring(1, @shfl_xor_sync(mask, val, lanemask, width)) == "ParallelStencil.ParallelKernel.@shfl_xor_sync mask val lanemask width"
            @test @prettystring(1, @vote_any_sync(mask, predicate)) == "ParallelStencil.ParallelKernel.@vote_any_sync mask predicate"
            @test @prettystring(1, @vote_all_sync(mask, predicate)) == "ParallelStencil.ParallelKernel.@vote_all_sync mask predicate"
            @test @prettystring(1, @vote_ballot_sync(mask, predicate)) == "ParallelStencil.ParallelKernel.@vote_ballot_sync mask predicate"
        end

        @testset "CPU semantic smoke tests" begin
            @static if @iscpu($package)
                N = 8
                A = @rand(N)
                P = [isfinite(A[i]) && (A[i] > zero($FloatDefault)) for i in 1:N]
                Bout_any    = Vector{Bool}(undef, N)
                Bout_all    = Vector{Bool}(undef, N)
                Bout_ballot = Vector{UInt64}(undef, N)
                Bshfl       = similar(A)
                Bshfl_up    = similar(A)
                Bshfl_down  = similar(A)
                Bshfl_xor   = similar(A)

                @parallel_indices (ix) function kernel_semantics!(Bout_any, Bout_all, Bout_ballot, Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, A, P)
                    m = @active_mask()
                    w = @warpsize()
                    l = @laneid()
                    @test w == 1
                    @test l == 1
                    Bshfl[ix]      = @shfl_sync(m, A[ix], l)
                    Bshfl_up[ix]   = @shfl_up_sync(m, A[ix], 1)
                    Bshfl_down[ix] = @shfl_down_sync(m, A[ix], 1)
                    Bshfl_xor[ix]  = @shfl_xor_sync(m, A[ix], 1)
                    pa = P[ix]
                    Bout_any[ix]    = @vote_any_sync(m, pa)
                    Bout_all[ix]    = @vote_all_sync(m, pa)
                    Bout_ballot[ix] = @vote_ballot_sync(m, pa)
                    return
                end
                @parallel (1:N) kernel_semantics!(Bout_any, Bout_all, Bout_ballot, Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, A, P)

                @test all(Bshfl .== A)
                @test all(Bshfl_up .== A)
                @test all(Bshfl_down .== A)
                @test all(Bshfl_xor .== A)
                @test Bout_any == P
                @test Bout_all == P
                @test Bout_ballot == map(p -> p ? UInt64(0x1) : UInt64(0x0), P)
            end
        end

        @reset_parallel_stencil()
    end
    ))
end == nothing || true;