using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_THREADS, PKG_POLYESTER, PKG_KERNELABSTRACTIONS
import ParallelStencil.ParallelKernel: @require, @prettystring, @iscpu, @select_hardware, @current_hardware, handle
import ParallelStencil.ParallelKernel: checknoargs, checkargs_sharedMem, Dim3
using ParallelStencil.ParallelKernel.Exceptions
TEST_PACKAGES = SUPPORTED_PACKAGES
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
@static if PKG_METAL in TEST_PACKAGES
    import Metal # Import also on non-Apple systems to test macro expansions
    if !Metal.functional() TEST_PACKAGES = filter!(x->x≠PKG_METAL, TEST_PACKAGES) end
end
@static if PKG_KERNELABSTRACTIONS in TEST_PACKAGES
    import KernelAbstractions
    if !KernelAbstractions.functional(KernelAbstractions.CPU()) TEST_PACKAGES = filter!(x->x≠PKG_KERNELABSTRACTIONS, TEST_PACKAGES) end
end
@static if PKG_POLYESTER in TEST_PACKAGES
    import Polyester
end
Base.retry_load_extensions() # Potentially needed to load the extensions after the packages have been filtered.


macro expr_allocated(ex)
    expanded = Base.macroexpand(__module__, ex; recursive=true)
    quote
        # Warm-up evaluation to exclude first-call setup allocations
        let
            $(esc(expanded))
        end
        @allocated begin
            $(esc(expanded))
        end
    end
end


@static for package in TEST_PACKAGES
    FloatDefault = (package == PKG_METAL) ? Float32 : Float64 # Metal does not support Float64
    
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @testset "1. kernel language macros" begin
            @require !@is_initialized()
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized()
            @testset "mapping to package" begin
                @static if $package == $PKG_CUDA
                    @test @prettystring(1, @gridDim()) == "CUDA.gridDim()"
                    @test @prettystring(1, @blockIdx()) == "CUDA.blockIdx()"
                    @test @prettystring(1, @blockDim()) == "CUDA.blockDim()"
                    @test @prettystring(1, @threadIdx()) == "CUDA.threadIdx()"
                    @test @prettystring(1, @sync_threads()) == "CUDA.sync_threads()"
                    @test @prettystring(1, @sharedMem($FloatDefault, (2,3))) == "CUDA.@cuDynamicSharedMem $(nameof($FloatDefault)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "CUDA.@cushow"
                    # @test @prettystring(1, @pk_println()) == "CUDA.@cuprintln"
                elseif $package == $PKG_AMDGPU
                    @test @prettystring(1, @gridDim()) == "AMDGPU.gridGroupDim()"
                    @test @prettystring(1, @blockIdx()) == "AMDGPU.workgroupIdx()"
                    @test @prettystring(1, @blockDim()) == "AMDGPU.workgroupDim()"
                    @test @prettystring(1, @threadIdx()) == "AMDGPU.workitemIdx()"
                    @test @prettystring(1, @sync_threads()) == "AMDGPU.sync_workgroup()"
                    # @test @prettystring(1, @sharedMem($FloatDefault, (2,3))) == ""    #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_show()) == "CUDA.@cushow"        #TODO: not yet supported for AMDGPU
                    # @test @prettystring(1, @pk_println()) == "AMDGPU.@rocprintln"
                elseif $package == $PKG_METAL
                    @test @prettystring(1, @gridDim()) == "Metal.threadgroups_per_grid_3d()"
                    @test @prettystring(1, @blockIdx()) == "Metal.threadgroup_position_in_grid_3d()"
                    @test @prettystring(1, @blockDim()) == "Metal.threads_per_threadgroup_3d()"
                    @test @prettystring(1, @threadIdx()) == "Metal.thread_position_in_threadgroup_3d()"
                    @test @prettystring(1, @sync_threads()) == "Metal.threadgroup_barrier(; flag = Metal.MemoryFlagThreadGroup)"
                    @test @prettystring(1, @sharedMem($FloatDefault, (2,3))) == "ParallelStencil.ParallelKernel.@sharedMem_metal $(nameof($FloatDefault)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "Metal.@mtlshow"        #TODO: not yet supported for Metal
                    # @test @prettystring(1, @pk_println()) == "Metal.@mtlprintln"  #TODO: not yet supported for Metal
                elseif $package == $PKG_KERNELABSTRACTIONS
                    call = @prettystring(1, @gridDim())
                    @test occursin("ParallelStencil.ParallelKernel.@gridDim_kernelabstractions", call)

                    call = @prettystring(2, @gridDim())
                    @test occursin("@ndrange", call)
                    @test occursin("@groupsize", call)

                    call = @prettystring(1, @blockIdx())
                    @test occursin("ParallelStencil.ParallelKernel.@blockIdx_kernelabstractions", call)

                    call = @prettystring(2, @blockIdx())
                    @test occursin("index_group_ntuple", call)

                    call = @prettystring(1, @blockDim())
                    @test occursin("ParallelStencil.ParallelKernel.@blockDim_kernelabstractions", call)

                    call = @prettystring(2, @blockDim())
                    @test occursin("@groupsize", call)

                    call = @prettystring(1, @threadIdx())
                    @test occursin("ParallelStencil.ParallelKernel.@threadIdx_kernelabstractions", call)

                    call = @prettystring(2, @threadIdx())
                    @test occursin("index_local_ntuple", call)

                    call = @prettystring(1, @sync_threads())
                    @test occursin("KernelAbstractions.@synchronize", call)

                    call = @prettystring(1, @sharedMem($FloatDefault, (2,3)))
                    @test occursin("ParallelStencil.ParallelKernel.@sharedMem_kernelabstractions", call)

                    call = @prettystring(2, @sharedMem($FloatDefault, (2,3)))
                    @test occursin("KernelAbstractions.@localmem", call)
                    @test occursin("$(nameof($FloatDefault))", call)
                elseif @iscpu($package)
                    @test @prettystring(1, @gridDim()) == "ParallelStencil.ParallelKernel.@gridDim_cpu"
                    @test @prettystring(1, @blockIdx()) == "ParallelStencil.ParallelKernel.@blockIdx_cpu"
                    @test @prettystring(1, @blockDim()) == "ParallelStencil.ParallelKernel.@blockDim_cpu"
                    @test @prettystring(1, @threadIdx()) == "ParallelStencil.ParallelKernel.@threadIdx_cpu"
                    @test @prettystring(1, @sync_threads()) == "ParallelStencil.ParallelKernel.@sync_threads_cpu"
                    @test @prettystring(1, @sharedMem($FloatDefault, (2,3))) == "ParallelStencil.ParallelKernel.@sharedMem_cpu $(nameof($FloatDefault)) (2, 3)"
                    # @test @prettystring(1, @pk_show()) == "Base.@show"
                    # @test @prettystring(1, @pk_println()) == "Base.println()"
                end;
            end;
            @testset "mapping to package (internal macros)" begin
                @static if $package == $PKG_THREADS
                    @test @prettystring(1, ParallelStencil.ParallelKernel.@threads()) == "Base.Threads.@threads"
                elseif $package == $PKG_POLYESTER
                    @test @prettystring(1, ParallelStencil.ParallelKernel.@threads()) == "Polyester.@batch"
                end;
            end;
            @testset "Warp level primitives" begin
                @testset "Parse-time direct call mapping" begin
                    # Common test variables used in macro expansions
                    mask      = UInt64(0xffff_ffff_ffff_ffff)
                    mask32    = UInt32(0xffff_ffff)
                    val       = one($FloatDefault)
                    lane      = 1
                    width     = 32
                    delta     = 1
                    lane_mask = 1
                    predicate = true

                    @static if $package == $PKG_CUDA
                        @test @prettystring(1, @warpsize()) == "CUDA.warpsize()"
                        @test @prettystring(1, @laneid())   == "CUDA.laneid() + 1"
                        @test @prettystring(1, @active_mask()) == "CUDA.active_mask()"

                        @test @prettystring(1, @shfl_sync(mask32, val, lane)) == "CUDA.shfl_sync(mask32, val, lane)"
                        @test @prettystring(1, @shfl_sync(mask32, val, lane, width)) == "CUDA.shfl_sync(mask32, val, lane, width)"
                        @test @prettystring(1, @shfl_up_sync(mask32, val, delta)) == "CUDA.shfl_up_sync(mask32, val, delta)"
                        @test @prettystring(1, @shfl_up_sync(mask32, val, delta, width)) == "CUDA.shfl_up_sync(mask32, val, delta, width)"
                        @test @prettystring(1, @shfl_down_sync(mask32, val, delta)) == "CUDA.shfl_down_sync(mask32, val, delta)"
                        @test @prettystring(1, @shfl_down_sync(mask32, val, delta, width)) == "CUDA.shfl_down_sync(mask32, val, delta, width)"
                        @test @prettystring(1, @shfl_xor_sync(mask32, val, lane_mask)) == "CUDA.shfl_xor_sync(mask32, val, lane_mask)"
                        @test @prettystring(1, @shfl_xor_sync(mask32, val, lane_mask, width)) == "CUDA.shfl_xor_sync(mask32, val, lane_mask, width)"

                        @test @prettystring(1, @vote_any_sync(mask32, predicate))   == "CUDA.vote_any_sync(mask32, predicate)"
                        @test @prettystring(1, @vote_all_sync(mask32, predicate))   == "CUDA.vote_all_sync(mask32, predicate)"
                        @test @prettystring(1, @vote_ballot_sync(mask32, predicate)) == "CUDA.vote_ballot_sync(mask32, predicate)"

                    elseif $package == $PKG_AMDGPU
                        @test @prettystring(1, @warpsize()) == "AMDGPU.Device.wavefrontsize()"
                        @test @prettystring(1, @laneid())   == "unsafe_trunc(Cint, AMDGPU.Device.activelane()) + Cint(1)"
                        @test @prettystring(1, @active_mask()) == "AMDGPU.Device.activemask()"

                        @test @prettystring(1, @shfl_sync(mask, val, lane)) == "AMDGPU.Device.shfl_sync(UInt64(mask), val, unsafe_trunc(Cint, lane) - Cint(1))"
                        @test @prettystring(1, @shfl_sync(mask, val, lane, width)) == "AMDGPU.Device.shfl_sync(UInt64(mask), val, unsafe_trunc(Cint, lane) - Cint(1), unsafe_trunc(Cuint, width))"
                        @test @prettystring(1, @shfl_up_sync(mask, val, delta)) == "AMDGPU.Device.shfl_up_sync(UInt64(mask), val, unsafe_trunc(Cint, delta))"
                        @test @prettystring(1, @shfl_up_sync(mask, val, delta, width)) == "AMDGPU.Device.shfl_up_sync(UInt64(mask), val, unsafe_trunc(Cint, delta), unsafe_trunc(Cuint, width))"
                        @test @prettystring(1, @shfl_down_sync(mask, val, delta)) == "AMDGPU.Device.shfl_down_sync(UInt64(mask), val, unsafe_trunc(Cint, delta))"
                        @test @prettystring(1, @shfl_down_sync(mask, val, delta, width)) == "AMDGPU.Device.shfl_down_sync(UInt64(mask), val, unsafe_trunc(Cint, delta), unsafe_trunc(Cuint, width))"
                        @test @prettystring(1, @shfl_xor_sync(mask, val, lane_mask)) == "AMDGPU.Device.shfl_xor_sync(UInt64(mask), val, unsafe_trunc(Cint, lane_mask) - Cint(1))"
                        @test @prettystring(1, @shfl_xor_sync(mask, val, lane_mask, width)) == "AMDGPU.Device.shfl_xor_sync(UInt64(mask), val, unsafe_trunc(Cint, lane_mask) - Cint(1), unsafe_trunc(Cuint, width))"

                        @test @prettystring(1, @vote_any_sync(mask, predicate))   == "AMDGPU.Device.any_sync(UInt64(mask), predicate)"
                        @test @prettystring(1, @vote_all_sync(mask, predicate))   == "AMDGPU.Device.all_sync(UInt64(mask), predicate)"
                        @test @prettystring(1, @vote_ballot_sync(mask, predicate)) == "AMDGPU.Device.ballot_sync(UInt64(mask), predicate)"

                    elseif $package == $PKG_METAL
                        @test @prettystring(1, @warpsize()) == "Metal.threads_per_simdgroup()"
                        @test @prettystring(1, @laneid())   == "unsafe_trunc(Cint, Metal.thread_index_in_simdgroup()) + Cint(1)"

                    elseif @iscpu($package)
                        @test @prettystring(1, @warpsize())     == "ParallelStencil.ParallelKernel.warpsize_cpu()"
                        @test @prettystring(1, @laneid())       == "ParallelStencil.ParallelKernel.laneid_cpu()"
                        @test @prettystring(1, @active_mask())  == "ParallelStencil.ParallelKernel.active_mask_cpu()"

                        @test @prettystring(1, @shfl_sync(mask, val, lane)) == "ParallelStencil.ParallelKernel.shfl_sync_cpu(mask, val, Int64(lane) - Int64(1))"
                        @test @prettystring(1, @shfl_sync(mask, val, lane, width)) == "ParallelStencil.ParallelKernel.shfl_sync_cpu(mask, val, Int64(lane) - Int64(1), Int64(width))"
                        @test @prettystring(1, @shfl_up_sync(mask, val, delta)) == "ParallelStencil.ParallelKernel.shfl_up_sync_cpu(mask, val, Int64(delta))"
                        @test @prettystring(1, @shfl_up_sync(mask, val, delta, width)) == "ParallelStencil.ParallelKernel.shfl_up_sync_cpu(mask, val, Int64(delta), Int64(width))"
                        @test @prettystring(1, @shfl_down_sync(mask, val, delta)) == "ParallelStencil.ParallelKernel.shfl_down_sync_cpu(mask, val, Int64(delta))"
                        @test @prettystring(1, @shfl_down_sync(mask, val, delta, width)) == "ParallelStencil.ParallelKernel.shfl_down_sync_cpu(mask, val, Int64(delta), Int64(width))"
                        @test @prettystring(1, @shfl_xor_sync(mask, val, lane_mask)) == "ParallelStencil.ParallelKernel.shfl_xor_sync_cpu(mask, val, Int64(lane_mask) - Int64(1))"
                        @test @prettystring(1, @shfl_xor_sync(mask, val, lane_mask, width)) == "ParallelStencil.ParallelKernel.shfl_xor_sync_cpu(mask, val, Int64(lane_mask) - Int64(1), Int64(width))"

                        @test @prettystring(1, @vote_any_sync(mask, predicate))   == "ParallelStencil.ParallelKernel.vote_any_sync_cpu(mask, predicate)"
                        @test @prettystring(1, @vote_all_sync(mask, predicate))   == "ParallelStencil.ParallelKernel.vote_all_sync_cpu(mask, predicate)"
                        @test @prettystring(1, @vote_ballot_sync(mask, predicate)) == "ParallelStencil.ParallelKernel.vote_ballot_sync_cpu(mask, predicate)"
                    end
                end;
                @testset "CPU zero overhead" begin
                    @static if @iscpu($package)
                        # Use stable literal arguments to exercise CPU code paths
                        mask      = UInt64(0x1)
                        valf      = one($FloatDefault)
                        lane      = 1
                        width     = 1
                        delta     = 1
                        lanemask  = 1
                        predicate = true

                        @test @expr_allocated(@warpsize())    == 0
                        @test @expr_allocated(@laneid())      == 0
                        @test @expr_allocated(@active_mask()) == 0

                        @test @expr_allocated(@shfl_sync(mask, valf, lane))            == 0
                        @test @expr_allocated(@shfl_sync(mask, valf, lane, width))     == 0
                        @test @expr_allocated(@shfl_up_sync(mask, valf, delta))        == 0
                        @test @expr_allocated(@shfl_up_sync(mask, valf, delta, width)) == 0
                        @test @expr_allocated(@shfl_down_sync(mask, valf, delta))      == 0
                        @test @expr_allocated(@shfl_down_sync(mask, valf, delta, width)) == 0
                        @test @expr_allocated(@shfl_xor_sync(mask, valf, lanemask))    == 0
                        @test @expr_allocated(@shfl_xor_sync(mask, valf, lanemask, width)) == 0

                        @test @expr_allocated(@vote_any_sync(mask, predicate))    == 0
                        @test @expr_allocated(@vote_all_sync(mask, predicate))    == 0
                        @test @expr_allocated(@vote_ballot_sync(mask, predicate)) == 0
                    end
                end;
                @testset "Semantic smoke tests" begin
                    @static if @iscpu($package)
                        N = 8
                        A  = @rand(N)
                        P  = [isfinite(A[i]) && (A[i] > zero($FloatDefault)) for i in 1:N]  # simple predicate
                        Bout_any    = Vector{Bool}(undef, N)
                        Bout_all    = Vector{Bool}(undef, N)
                        Bout_ballot = Vector{UInt64}(undef, N)
                        Bshfl       = similar(A)
                        Bshfl_up    = similar(A)
                        Bshfl_down  = similar(A)
                        Bshfl_xor   = similar(A)
                        Bwarpsize   = Vector{Int}(undef, N)
                        Blaneid     = Vector{Int}(undef, N)

                        @parallel_indices (ix) function kernel_semantics!(Bout_any, Bout_all, Bout_ballot, Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, Bwarpsize, Blaneid, A, P)
                            m = @active_mask()
                            w = @warpsize()
                            l = @laneid()
                            # store values for verification outside kernel
                            Bwarpsize[ix] = w
                            Blaneid[ix] = l
                            # shuffle identities
                            Bshfl[ix]      = @shfl_sync(m, A[ix], l)
                            Bshfl_up[ix]   = @shfl_up_sync(m, A[ix], 1)
                            Bshfl_down[ix] = @shfl_down_sync(m, A[ix], 1)
                            Bshfl_xor[ix]  = @shfl_xor_sync(m, A[ix], 1)
                            # votes
                            pa = P[ix]
                            Bout_any[ix]   = @vote_any_sync(m, pa)
                            Bout_all[ix]   = @vote_all_sync(m, pa)
                            Bout_ballot[ix] = @vote_ballot_sync(m, pa)
                            return
                        end
                        @parallel (1:N) kernel_semantics!(Bout_any, Bout_all, Bout_ballot, Bshfl, Bshfl_up, Bshfl_down, Bshfl_xor, Bwarpsize, Blaneid, A, P)

                        # basic invariants under CPU model
                        @test all(Bwarpsize .== 1)
                        @test all(Blaneid .== 1)
                        @test all(Bshfl .== A)
                        @test all(Bshfl_up .== A)
                        @test all(Bshfl_down .== A)
                        @test all(Bshfl_xor .== A)
                        @test Bout_any == P
                        @test Bout_all == P
                        @test Bout_ballot == map(p -> p ? UInt64(0x1) : UInt64(0x0), P)
                    end
                end;
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (1D)" begin
                @static if $package == $PKG_THREADS
                    A  = @zeros(4)
                    @parallel_indices (ix) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 1, 1)
                        @test @blockIdx() == Dim3(ix-2, 1, 1)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (2D)" begin
                @static if $package == $PKG_THREADS
                    A  = @zeros(4, 5)
                    @parallel_indices (ix,iy) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 3, 1)
                        @test @blockIdx() == Dim3(ix-2, iy-1, 1)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4, 2:4) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "@gridDim, @blockIdx, @blockDim, @threadIdx (3D)" begin
                @static if $package == $PKG_THREADS
                    A  = @zeros(4, 5, 6)
                    @parallel_indices (ix,iy,iz) function test_macros!(A)
                        @test @gridDim() == Dim3(2, 3, 6)
                        @test @blockIdx() == Dim3(ix-2, iy-1, iz)
                        @test @blockDim() == Dim3(1, 1, 1)
                        @test @threadIdx() == Dim3(1, 1, 1)
                        return
                    end
                    @parallel (3:4, 2:4, 1:6) test_macros!(A);
                    @test true # @gridDim test succeeded if this line is reached (the above tests within the kernel are not captured if they succeed, only if they fail...; alternative test implementations would be of course possible, but would be more complex).
                    @test true # @blockIdx ...
                    @test true # @blockDim ...
                    @test true # @threadIdx ...
                end
            end;
            @testset "sync_threads" begin
                @static if @iscpu($package)
                    @test @prettystring(ParallelStencil.ParallelKernel.@sync_threads_cpu()) == "begin\nend"
                end;
            end;
            @testset "shared memory (allocation)" begin
                @static if @iscpu($package)
                    @test typeof(@sharedMem($FloatDefault,(2,3))) == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3},   $FloatDefault, length((2,3)),   prod((2,3))}(undef))
                    @test typeof(@sharedMem(Bool,(2,3,4)))  == typeof(ParallelStencil.ParallelKernel.MArray{Tuple{2,3,4}, Bool,    length((2,3,4)), prod((2,3,4))}(undef))
                end;
            end;
            @testset "@sharedMem (1D)" begin
                @static if @iscpu($package)
                    A  = @rand(4)
                    B  = @zeros(4)
                    @parallel_indices (ix) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        A_l = @sharedMem(eltype(A), (@blockDim().x))
                        A_l[tx] = A[ix]
                        @sync_threads()
                        B[ix] = A_l[tx]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @testset "@sharedMem (2D)" begin
                @static if @iscpu($package)
                    A  = @rand(4,5)
                    B  = @zeros(4,5)
                    @parallel_indices (ix,iy) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        ty  = @threadIdx().y
                        A_l = @sharedMem(eltype(A), (@blockDim().x, @blockDim().y), 0*sizeof(eltype(A)))
                        A_l[tx,ty] = A[ix,iy]
                        @sync_threads()
                        B[ix,iy] = A_l[tx,ty]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @testset "@sharedMem (3D)" begin
                @static if @iscpu($package)
                    A  = @rand(4,5,6)
                    B  = @zeros(4,5,6)
                    @parallel_indices (ix,iy,iz) function memcopy!(B, A)
                        tx  = @threadIdx().x
                        ty  = @threadIdx().y
                        tz  = @threadIdx().z
                        A_l = @sharedMem(eltype(A), (@blockDim().x, @blockDim().y, @blockDim().z))
                        A_l[tx,ty,tz] = A[ix,iy,iz]
                        @sync_threads()
                        B[ix,iy,iz] = A_l[tx,ty,tz]
                        return
                    end
                    @parallel memcopy!(B, A);
                    @test B == A
                end
            end;
            @testset "@∀" begin
                expansion = @prettystring(1, @∀ i ∈ (x,z) @all(C.i) = @all(A.i) + @all(B.i))
                @test occursin("@all(C.x) = @all(A.x) + @all(B.x)", expansion)
                @test occursin("@all(C.z) = @all(A.z) + @all(B.z)", expansion)
                expansion = @prettystring(1, @∀ i ∈ (y,z) C.i[ix,iy,iz] = A.i[ix,iy,iz] + B.i[ix,iy,iz])
                @test occursin("C.y[ix, iy, iz] = A.y[ix, iy, iz] + B.y[ix, iy, iz]", expansion)
                @test occursin("C.z[ix, iy, iz] = A.z[ix, iy, iz] + B.z[ix, iy, iz]", expansion)
                expansion = @prettystring(1, @∀ (ij,i,j) ∈ ((xy,x,y), (xz,x,z), (yz,y,z)) @all(C.ij) = @all(A.i) + @all(B.j))
                @test occursin("@all(C.xy) = @all(A.x) + @all(B.y)", expansion)
                @test occursin("@all(C.xz) = @all(A.x) + @all(B.z)", expansion)
                @test occursin("@all(C.yz) = @all(A.y) + @all(B.z)", expansion)
                expansion = @prettystring(1, @∀ i ∈ 1:N-1 @all(C[i]) = @all(A[i]) + @all(B[i]))
                @test occursin("ntuple(Val(N - 1)) do i", expansion)
                @test occursin("@all(C[i]) = @all(A[i]) + @all(B[i])", expansion)
                expansion = @prettystring(1, @∀ i ∈ 2:N-1 C[i][ix,iy,iz] = A[i][ix,iy,iz] + B[i][ix,iy,iz])
                @test occursin("ntuple(Val(((N - 1) - 2) + 1)) do i", expansion)
                @test occursin("(C[(i + 2) - 1])[ix, iy, iz] = (A[(i + 2) - 1])[ix, iy, iz] + (B[(i + 2) - 1])[ix, iy, iz]", expansion)
            end;
            @reset_parallel_kernel()
        end;
        @testset "2. Exceptions" begin
            @init_parallel_kernel($package, $FloatDefault)
            @require @is_initialized
            @testset "no arguments" begin
                @test_throws ArgumentError checknoargs(:(something));                                                   # Error: length(args) != 0
            end;
            @testset "arguments @sharedMem" begin
                @test_throws ArgumentError checkargs_sharedMem();                                                        # Error: isempty(args)
                @test_throws ArgumentError checkargs_sharedMem(:(something));                                            # Error: length(args) != 2
                @test_throws ArgumentError checkargs_sharedMem(:(something), :(something), :(something), :(something));  # Error: length(args) != 2
            end;
            @reset_parallel_kernel()
        end;
    end;
))

end == nothing || true;
