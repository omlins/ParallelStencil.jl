using Test
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, select_hardware, current_hardware
import ParallelStencil: @require, @prettystring, @iscpu

const PKG_KERNELABSTRACTIONS = hasproperty(ParallelStencil, :PKG_KERNELABSTRACTIONS) ? ParallelStencil.PKG_KERNELABSTRACTIONS : Symbol(:KernelAbstractions)

TEST_PACKAGES = collect(SUPPORTED_PACKAGES)
if PKG_KERNELABSTRACTIONS ∉ TEST_PACKAGES
    push!(TEST_PACKAGES, PKG_KERNELABSTRACTIONS)
end
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
    @static if Sys.isapple()
        import Metal
        if !Metal.functional()
            TEST_PACKAGES = filter!(x -> x ≠ PKG_METAL, TEST_PACKAGES)
        end
    else
        TEST_PACKAGES = filter!(x -> x ≠ PKG_METAL, TEST_PACKAGES)
    end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    if Base.find_package("KernelAbstractions") === nothing
        TEST_PACKAGES = filter!(x -> x ≠ PKG_KERNELABSTRACTIONS, TEST_PACKAGES)
    else
        import KernelAbstractions
    end
end
Base.retry_load_extensions()

kernelabstractions_gpu_symbols() = Symbol[]

@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    function kernelabstractions_gpu_symbols()
        symbols = Symbol[]
        if isdefined(@__MODULE__, :CUDA) && CUDA.functional()
            push!(symbols, :gpu_cuda)
        end
        if isdefined(@__MODULE__, :AMDGPU) && AMDGPU.functional()
            push!(symbols, :gpu_amd)
        end
        if isdefined(@__MODULE__, :Metal)
            if Sys.isapple() && Metal.functional()
                push!(symbols, :gpu_metal)
            end
        end
        return symbols
    end
end

@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL || package == PKG_KERNELABSTRACTIONS) ? Float32 : Float64

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

            if $package == $PKG_KERNELABSTRACTIONS
                select_hardware(:cpu)
                @test current_hardware() == :cpu
                call = @prettystring(1, @gridDim())
                @test occursin("ParallelStencil.ParallelKernel.@gridDim", call)
                call = @prettystring(1, @blockIdx())
                @test occursin("ParallelStencil.ParallelKernel.@blockIdx", call)
                call = @prettystring(1, @blockDim())
                @test occursin("ParallelStencil.ParallelKernel.@blockDim", call)
                call = @prettystring(1, @threadIdx())
                @test occursin("ParallelStencil.ParallelKernel.@threadIdx", call)
                call = @prettystring(1, @sync_threads())
                @test occursin("ParallelStencil.ParallelKernel.@sync_threads", call)
                call = @prettystring(1, @sharedMem(T, dims))
                @test occursin("ParallelStencil.ParallelKernel.@sharedMem", call)

                for symbol in kernelabstractions_gpu_symbols()
                    select_hardware(symbol)
                    call = @prettystring(1, @gridDim())
                    if symbol == :gpu_cuda
                        @test call == "CUDA.gridDim()"
                        @test @prettystring(1, @blockIdx()) == "CUDA.blockIdx()"
                        @test @prettystring(1, @blockDim()) == "CUDA.blockDim()"
                        @test @prettystring(1, @threadIdx()) == "CUDA.threadIdx()"
                        @test @prettystring(1, @sync_threads()) == "CUDA.sync_threads()"
                        @test @prettystring(1, @sharedMem(T, dims)) == "CUDA.@cuDynamicSharedMem T dims"
                    elseif symbol == :gpu_amd
                        @test call == "AMDGPU.gridGroupDim()"
                        @test @prettystring(1, @blockIdx()) == "AMDGPU.workgroupIdx()"
                        @test @prettystring(1, @blockDim()) == "AMDGPU.workgroupDim()"
                        @test @prettystring(1, @threadIdx()) == "AMDGPU.workitemIdx()"
                        @test @prettystring(1, @sync_threads()) == "AMDGPU.sync_workgroup()"
                        # @test @prettystring(1, @sharedMem(T, dims)) == ""    #TODO: not yet supported for AMDGPU
                    elseif symbol == :gpu_metal
                        @test call == "Metal.threadgroups_per_grid_3d()"
                        @test @prettystring(1, @blockIdx()) == "Metal.threadgroup_position_in_grid_3d()"
                        @test @prettystring(1, @blockDim()) == "Metal.threads_per_threadgroup_3d()"
                        @test @prettystring(1, @threadIdx()) == "Metal.thread_position_in_threadgroup_3d()"
                        @test @prettystring(1, @sync_threads()) == "Metal.threadgroup_barrier(; flag = Metal.MemoryFlagThreadGroup)"
                        @test @prettystring(1, @sharedMem(T, dims)) == "ParallelStencil.ParallelKernel.@sharedMem_metal T dims"
                    end
                    @test current_hardware() == symbol
                end
                select_hardware(:cpu)
                @test current_hardware() == :cpu
                @test_throws ArgumentError select_hardware(:unsupported_hardware_symbol)
            end

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
            @static if @iscpu($package) || $package == $PKG_KERNELABSTRACTIONS
                if $package == $PKG_KERNELABSTRACTIONS
                    select_hardware(:cpu)
                end
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
                if $package == $PKG_KERNELABSTRACTIONS
                    @test current_hardware() == :cpu
                end
            end
        end

        @reset_parallel_stencil()
    end
    ))
end == nothing || true;