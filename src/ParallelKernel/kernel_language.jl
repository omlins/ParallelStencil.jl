#NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1).
# The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
# "Shared" memory (which will belong to the only thread in the block) will be allocated directly in the loop (i.e., for each parallel index) as local variable.


## MACROS

##
const GRIDDIM_DOC = """
    @gridDim()

Return the grid size (or "dimension") in x, y and z dimension. The grid size in a specific dimension is commonly retrieved directly as in this example in x dimension: `@gridDim().x`.
"""
@doc GRIDDIM_DOC
macro gridDim(args...) check_initialized(__module__); checknoargs(args...); esc(gridDim(__module__, args...)); end


##
const BLOCKIDX_DOC = """
    @blockIdx()

Return the block ID in x, y and z dimension within the grid. The block ID in a specific dimension is commonly retrieved directly as in this example in x dimension: `@blockIdx().x`.
"""
@doc BLOCKIDX_DOC
macro blockIdx(args...) check_initialized(__module__); checknoargs(args...); esc(blockIdx(__module__, args...)); end


##
const BLOCKDIM_DOC = """
    @blockDim()

Return the block size (or "dimension") in x, y and z dimension. The block size in a specific dimension is commonly retrieved directly as in this example in x dimension: `@blockDim().x`.
"""
@doc BLOCKDIM_DOC
macro blockDim(args...) check_initialized(__module__); checknoargs(args...); esc(blockDim(__module__, args...)); end


##
const THREADIDX_DOC = """
    @threadIdx()

Return the thread ID in x, y and z dimension within the block. The thread ID in a specific dimension is commonly retrieved directly as in this example in x dimension: `@threadIdx().x`.
"""
@doc THREADIDX_DOC
macro threadIdx(args...) check_initialized(__module__); checknoargs(args...); esc(threadIdx(__module__, args...)); end


##
const SYNCTHREADS_DOC = """
    @sync_threads()

Synchronize the threads of the block: wait until all threads in the block have reached this point and all global and shared memory accesses made by these threads prior to the `sync_threads()` call are visible to all threads in the block.
"""
@doc SYNCTHREADS_DOC
macro sync_threads(args...) check_initialized(__module__); checknoargs(args...); esc(sync_threads(__module__, args...)); end


##
const SHAREDMEM_DOC = """
    @sharedMem(T, dims, offset::Integer=0)

Create an array that is *shared* between the threads of a block (i.e. accessible only by the threads of a same block), with element type `T` and size specified by `dims`. 
When multiple shared memory arrays are created within a kernel, then all arrays except for the first one typically need to define the `offset` to the base shared memory pointer in bytes (note that the CPU and AMDGPU implementation do not require the `offset` and will simply ignore it when present).

!!! note "Note"
    The amount of shared memory needs to be specified when launching the kernel (keyword argument `shmem`).
"""
@doc SHAREDMEM_DOC
macro sharedMem(args...) check_initialized(__module__); checkargs_sharedMem(args...); esc(sharedMem(__module__, args...)); end


##
const PKSHOW_DOC = """
    @pk_show(...)

Call a macro analogue to `Base.@show`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@show for Threads or Polyester and CUDA.@cushow for CUDA).
"""
@doc PKSHOW_DOC
macro pk_show(args...) check_initialized(__module__); esc(pk_show(__module__, args...)); end


##
const PKPRINTLN_DOC = """
    @pk_println(...)

Call a macro analogue to `Base.@println`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@println for Threads or Polyester and CUDA.@cuprintln for CUDA).
"""
@doc PKPRINTLN_DOC
macro pk_println(args...) check_initialized(__module__); esc(pk_println(__module__, args...)); end


##
const WARPSIZE_DOC = """
    @warpsize() -> Int

Return the logical warp / wavefront / SIMD-group width in threads for the active backend.  CUDA returns 32. AMD GPUs return the hardware wavefront size (typically 64 or 32). Metal returns the device `threadExecutionWidth`. CPU backend returns 1.  Guaranteed constant for the lifetime of the kernel invocation. Use this value (not a hard‑coded constant) for portable intra-warp algorithms.
"""
@doc WARPSIZE_DOC
macro warpsize(args...) check_initialized(__module__); checknoargs(args...); esc(warpsize(__module__, args...)); end


##
const LANEID_DOC = """
    @laneid() -> Int

Return the 1-based logical lane index in the current warp (range: 1:warpsize()).  For CUDA this is `CUDA.laneid()+1` internally; for backends with 0-based hardware lane numbering the abstraction adds 1.  CPU backend always returns 1.
"""
@doc LANEID_DOC
macro laneid(args...) check_initialized(__module__); checknoargs(args...); esc(laneid(__module__, args...)); end


##
const ACTIVE_MASK_DOC = """
    @active_mask() -> Unsigned

Return a bit mask of currently active (non-exited, converged) lanes in the caller's warp.  Bit (laneid()-1) corresponds to that logical lane.  CUDA returns a 32-bit value; AMD returns a 64-bit value.  Absent (throws) on Metal if not supported; CPU returns UInt64(0x1).
"""
@doc ACTIVE_MASK_DOC
macro active_mask(args...) check_initialized(__module__); checknoargs(args...); esc(active_mask(__module__, args...)); end


##
const SHFL_SYNC_DOC = """
    @shfl_sync(mask::Unsigned, val, lane::Integer)
    @shfl_sync(mask::Unsigned, val, lane::Integer, width::Integer)

Return the value of `val` from the source lane `lane` (1-based) among lanes named in `mask`.  Optional `width` (power of two, 1 <= width <= warpsize()) logically partitions the warp into independent contiguous sub-groups each behaving as a mini-warp with lanes numbered 1:width.  The source lane index is resolved modulo `width`.  All participating lanes must supply identical `mask`, `lane`, and (if present) `width`.  `val` may be any isbits type; larger composite isbits values are shuffled by decomposition into supported word sizes.  CPU backend returns `val` unchanged.
"""
@doc SHFL_SYNC_DOC
macro shfl_sync(args...) check_initialized(__module__); checkargs_shfl_sync(args...); esc(shfl_sync(__module__, args...)); end


##
const SHFL_UP_SYNC_DOC = """
    @shfl_up_sync(mask::Unsigned, val, delta::Integer)
    @shfl_up_sync(mask::Unsigned, val, delta::Integer, width::Integer)

Shift `val` up by `delta` lanes within each logical partition (width semantics as in `shfl_sync`).  Lanes with no valid upstream partner retain their original `val`.  `delta >= 0`.  CPU backend returns `val` unchanged.
"""
@doc SHFL_UP_SYNC_DOC
macro shfl_up_sync(args...) check_initialized(__module__); checkargs_shfl_up_down_xor(args...); esc(shfl_up_sync(__module__, args...)); end


##
const SHFL_DOWN_SYNC_DOC = """
    @shfl_down_sync(mask::Unsigned, val, delta::Integer)
    @shfl_down_sync(mask::Unsigned, val, delta::Integer, width::Integer)

Shift `val` down by `delta` lanes within each logical partition; lanes without a valid downstream partner retain their original `val`.  `delta >= 0`.  CPU backend returns `val` unchanged.
"""
@doc SHFL_DOWN_SYNC_DOC
macro shfl_down_sync(args...) check_initialized(__module__); checkargs_shfl_up_down_xor(args...); esc(shfl_down_sync(__module__, args...)); end


##
const SHFL_XOR_SYNC_DOC = """
    @shfl_xor_sync(mask::Unsigned, val, lane_mask::Integer)
    @shfl_xor_sync(mask::Unsigned, val, lane_mask::Integer, width::Integer)

Perform a butterfly (bitwise XOR) shuffle: each lane exchanges with the lane whose (laneid()-1) XOR `lane_mask` differs in the specified bits, constrained within each `width` partition if provided.  If the computed partner is outside the partition the calling lane's own `val` is returned.  CPU backend returns `val` unchanged.
"""
@doc SHFL_XOR_SYNC_DOC
macro shfl_xor_sync(args...) check_initialized(__module__); checkargs_shfl_up_down_xor(args...); esc(shfl_xor_sync(__module__, args...)); end


##
const VOTE_ANY_SYNC_DOC = """
    @vote_any_sync(mask::Unsigned, predicate::Bool) -> Bool

Evaluate `predicate` across all active lanes named in `mask`; return true if any lane's predicate is true.  Does not imply a memory fence.  CPU backend returns `predicate`.
"""
@doc VOTE_ANY_SYNC_DOC
macro vote_any_sync(args...) check_initialized(__module__); checkargs_vote(args...); esc(vote_any_sync(__module__, args...)); end


##
const VOTE_ALL_SYNC_DOC = """
    @vote_all_sync(mask::Unsigned, predicate::Bool) -> Bool

Evaluate `predicate` across all active lanes named in `mask`; return true only if every such lane's predicate is true.  No memory ordering implied.  CPU backend returns `predicate`.
"""
@doc VOTE_ALL_SYNC_DOC
macro vote_all_sync(args...) check_initialized(__module__); checkargs_vote(args...); esc(vote_all_sync(__module__, args...)); end


##
const VOTE_BALLOT_SYNC_DOC = """
    @vote_ballot_sync(mask::Unsigned, predicate::Bool) -> Unsigned

Return a bit mask aggregating `predicate` values for lanes named in `mask`: bit (laneid()-1) set iff that lane's predicate is true.  Width of result equals hardware warp mask width (32 for CUDA, 64 for AMD, CPU uses 64 with only bit 0 meaningful).  Caller may safely promote to `UInt64` for uniform handling; upper bits beyond hardware width are zero.  No memory ordering implied.
"""
@doc VOTE_BALLOT_SYNC_DOC
macro vote_ballot_sync(args...) check_initialized(__module__); checkargs_vote(args...); esc(vote_ballot_sync(__module__, args...)); end


##
const FORALL_DOC = """
    @∀ x ∈ X statement
    @∀ x in X statement
    @∀ x = X statement

Expand the `statement` for all `x` in `X`.

# Arguments
- `x`: the name of the variable the `statement` is to be expanded with.
- `X`: the set of names or range of values `x` is to be expanded over.
- `statement`: the statement to be expanded.

# Examples
    @∀ i ∈ (x,z) @all(C.i) = @all(A.i) + @all(B.i)                     # Equivalent to: @all(C.x) = @all(A.x) + @all(B.x)
                                                                       #                @all(C.z) = @all(A.z) + @all(B.z)

    @∀ i ∈ (y,z) C.i[ix,iy,iz] = A.i[ix,iy,iz] + B.i[ix,iy,iz]         # Equivalent to: C.y[ix,iy,iz] = A.y[ix,iy,iz] + B.y[ix,iy,iz]
                                                                       #                C.z[ix,iy,iz] = A.z[ix,iy,iz] + B.z[ix,iy,iz]

    @∀ (ij,i,j) ∈ ((xy,x,y), (xz,x,z), (yz,y,z)) @all(C.ij) = @all(A.i) + @all(B.j)  # Equivalent to: @all(C.xy) = @all(A.x) + @all(B.y)
                                                                                     #                @all(C.xz) = @all(A.x) + @all(B.z)
                                                                                     #                @all(C.yz) = @all(A.y) + @all(B.z)

    @∀ i ∈ 1:N @all(C[i]) = @all(A[i]) + @all(B[i])                    # Equivalent to: @all(C[1]) = @all(A[1]) + @all(B[1])
                                                                       #                @all(C[2]) = @all(A[2]) + @all(B[2])
                                                                       #                ...
                                                                       #                @all(C[N]) = @all(A[N]) + @all(B[N])

    @∀ i ∈ 1:N C[i][ix,iy,iz] = A[i][ix,iy,iz] + B[i][ix,iy,iz]        # Equivalent to: C[1][ix,iy,iz] = A[1][ix,iy,iz] + B[1][ix,iy,iz]
                                                                       #                C[2][ix,iy,iz] = A[2][ix,iy,iz] + B[2][ix,iy,iz]
                                                                       #                ...
                                                                       #                C[N][ix,iy,iz] = A[N][ix,iy,iz] + B[N][ix,iy,iz]

!!! note "Performance note"
    Besides enabling a concise notation for certain sets of equations, the `@∀` macro is also designed to replace for loops over a small range of values in compute kernels, leading often to better performance.

!!! note "Symbol ∀"
    The symbol `∀` can be obtained, e.g., in the Julia REPL or VSCode, by typing `\forall` and pressing `TAB`.
"""
@doc FORALL_DOC
macro ∀(args...) check_initialized(__module__); checkforallargs(args...); esc(∀(__module__, args...)); end


## INTERNAL MACROS

##
macro threads(args...) check_initialized(__module__); esc(threads(__module__, args...)); end


##
macro firstindex(args...) check_initialized(__module__); checkargs_begin_end(args...); esc(_firstindex(__module__, args...)); end


##
macro lastindex(args...) check_initialized(__module__); checkargs_begin_end(args...); esc(_lastindex(__module__, args...)); end


##
macro return_value(args...) check_initialized(__module__); checksinglearg(args...); esc(return_value(args...)); end


##
macro return_nothing(args...) check_initialized(__module__); checknoargs(args...); esc(return_nothing(args...)); end


## ARGUMENT CHECKS

function checknoargs(args...)
    if (length(args) != 0) @ArgumentError("no arguments allowed.") end
end

function checksinglearg(args...)
    if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
end

function checkargs_sharedMem(args...)
    if !(2 <= length(args) <= 3) @ArgumentError("wrong number of arguments.") end
end

function checkforallargs(args...)
    if !(length(args) == 2) @ArgumentError("wrong number of arguments.") end
    if !((args[1].head == :call && args[1].args[1] in [:∈, :in]) || args[1].head == :(=)) @ArgumentError("the first argument must be of the form `x ∈ X, `x in X` or `x = X`.") end
end

function checkargs_begin_end(args...)
    if !(2 <= length(args) <= 3) @ArgumentError("wrong number of arguments.") end
end

function checkargs_shfl_sync(args...)
    if !(3 <= length(args) <= 4) @ArgumentError("wrong number of arguments.") end
end

function checkargs_shfl_up_down_xor(args...)
    if !(3 <= length(args) <= 4) @ArgumentError("wrong number of arguments.") end
end

function checkargs_vote(args...)
    if !(length(args) == 2) @ArgumentError("wrong number of arguments.") end
end


## FUNCTIONS FOR INDEXING AND DIMENSIONS

function gridDim(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.gridDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.gridGroupDim($(args...)))
    elseif (package == PKG_METAL)   return :(Metal.threadgroups_per_grid_3d($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@gridDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockIdx(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation relies on the fact that ranges are always of type UnitRange. If this changes, then this function needs to be adapted.
    if     (package == PKG_CUDA)    return :(CUDA.blockIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupIdx($(args...)))
    elseif (package == PKG_METAL)   return :(Metal.threadgroup_position_in_grid_3d($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@blockIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockDim(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.blockDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupDim($(args...)))
    elseif (package == PKG_METAL)   return :(Metal.threads_per_threadgroup_3d($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@blockDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function threadIdx(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.threadIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workitemIdx($(args...)))
    elseif (package == PKG_METAL)   return :(Metal.thread_position_in_threadgroup_3d($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@threadIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SYNCHRONIZATION

function sync_threads(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
    if     (package == PKG_CUDA)    return :(CUDA.sync_threads($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.sync_workgroup($(args...)))
    elseif (package == PKG_METAL)   return :(Metal.threadgroup_barrier($(args...); flag=Metal.MemoryFlagThreadGroup))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@sync_threads_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION

function sharedMem(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cuDynamicSharedMem($(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.@sharedMem_amdgpu($(args...)))
    elseif (package == PKG_METAL)   return :(ParallelStencil.ParallelKernel.@sharedMem_metal($(args...)))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.@sharedMem_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

macro sharedMem_amdgpu(T, dims) esc(:(AMDGPU.@ROCDynamicLocalArray($T, $dims, false))) end

macro sharedMem_amdgpu(T, dims, offset) esc(:(ParallelStencil.ParallelKernel.@sharedMem_amdgpu($T, $dims))) end

macro sharedMem_metal(T, dims) :(Metal.MtlThreadGroupArray($T, $dims)); end

macro sharedMem_metal(T, dims, offset) esc(:(ParallelStencil.ParallelKernel.@sharedMem_metal($T, $dims))) end

## FUNCTIONS FOR PRINTING

function pk_show(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cushow($(args...)))
    elseif (package == PKG_AMDGPU)  @KeywordArgumentError("this functionality is not yet supported in AMDGPU.jl.")
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(Base.@show($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function pk_println(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cuprintln($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.@rocprintln($(args...)))
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(Base.println($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR WARP-LEVEL PRIMITIVES (backend mapping)

function warpsize(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.warpsize())
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.Device.wavefrontsize())
    elseif (package == PKG_METAL)   return :(Metal.threads_per_simdgroup())
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.warpsize_cpu())
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function laneid(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.laneid() + 1)
    elseif (package == PKG_AMDGPU)  return :(unsafe_trunc(Cint, AMDGPU.Device.activelane()) + Cint(1))
    elseif (package == PKG_METAL)   return :(unsafe_trunc(Cint, Metal.thread_index_in_simdgroup()) + Cint(1))
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.laneid_cpu())
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function active_mask(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.active_mask())
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.Device.activemask())
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.active_mask_cpu())
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function shfl_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)
        return :(CUDA.shfl_sync($(args...)))
    elseif (package == PKG_AMDGPU)
        if length(args) == 3
            # (mask, val, lane)
            return :(AMDGPU.Device.shfl_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])) - Cint(1)))
        else
            # (mask, val, lane, width)
            return :(AMDGPU.Device.shfl_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])) - Cint(1), unsafe_trunc(Cuint, $(args[4]))))
        end
    elseif (package == PKG_METAL)
        @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)
        if length(args) == 3
            return :(ParallelStencil.ParallelKernel.shfl_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])) - Int64(1)))
        else
            return :(ParallelStencil.ParallelKernel.shfl_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])) - Int64(1), Int64($(args[4]))))
        end
    else
        @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function shfl_up_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)
        return :(CUDA.shfl_up_sync($(args...)))
    elseif (package == PKG_AMDGPU)
        if length(args) == 3
            return :(AMDGPU.Device.shfl_up_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3]))))
        else
            return :(AMDGPU.Device.shfl_up_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])), unsafe_trunc(Cuint, $(args[4]))))
        end
    elseif (package == PKG_METAL)
        @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)
        if length(args) == 3
            return :(ParallelStencil.ParallelKernel.shfl_up_sync_cpu($(args[1]), $(args[2]), Int64($(args[3]))))
        else
            return :(ParallelStencil.ParallelKernel.shfl_up_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])), Int64($(args[4]))))
        end
    else
        @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function shfl_down_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)
        return :(CUDA.shfl_down_sync($(args...)))
    elseif (package == PKG_AMDGPU)
        if length(args) == 3
            return :(AMDGPU.Device.shfl_down_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3]))))
        else
            return :(AMDGPU.Device.shfl_down_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])), unsafe_trunc(Cuint, $(args[4]))))
        end
    elseif (package == PKG_METAL)
        @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)
        if length(args) == 3
            return :(ParallelStencil.ParallelKernel.shfl_down_sync_cpu($(args[1]), $(args[2]), Int64($(args[3]))))
        else
            return :(ParallelStencil.ParallelKernel.shfl_down_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])), Int64($(args[4]))))
        end
    else
        @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function shfl_xor_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)
        return :(CUDA.shfl_xor_sync($(args...)))
    elseif (package == PKG_AMDGPU)
        if length(args) == 3
            return :(AMDGPU.Device.shfl_xor_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])) - Cint(1)))
        else
            return :(AMDGPU.Device.shfl_xor_sync(UInt64($(args[1])), $(args[2]), unsafe_trunc(Cint, $(args[3])) - Cint(1), unsafe_trunc(Cuint, $(args[4]))))
        end
    elseif (package == PKG_METAL)
        @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)
        if length(args) == 3
            return :(ParallelStencil.ParallelKernel.shfl_xor_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])) - Int64(1)))
        else
            return :(ParallelStencil.ParallelKernel.shfl_xor_sync_cpu($(args[1]), $(args[2]), Int64($(args[3])) - Int64(1), Int64($(args[4]))))
        end
    else
        @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function vote_any_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.vote_any_sync($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.Device.any_sync(UInt64($(args[1])), $(args[2])))
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.vote_any_sync_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function vote_all_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.vote_all_sync($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.Device.all_sync(UInt64($(args[1])), $(args[2])))
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.vote_all_sync_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function vote_ballot_sync(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.vote_ballot_sync($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.Device.ballot_sync(UInt64($(args[1])), $(args[2])))
    elseif (package == PKG_METAL)   @KeywordArgumentError("this functionality is not yet supported in Metal.jl.")
    elseif iscpu(package)           return :(ParallelStencil.ParallelKernel.vote_ballot_sync_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

## FUNCTIONS FOR MATH SYNTAX

function ∀(caller::Module, member_expr::Expr, statement::Union{Expr, Symbol})
    if !(@capture(member_expr, x_ ∈ X_) || @capture(member_expr, x_ in X_) || @capture(member_expr, x_ = X_)) @ArgumentError("the first argument must be of the form `x ∈ X, `x in X` or `x = X`.") end
    if @capture(X, a_:b_)
        niters    = (a == 1) ? b : :($b-$a+1)
        statement = (a == 1) ? statement : substitute(statement, x, :($x+$a-1))
        return quote
            ntuple(Val($niters)) do $x
                @inline
                $statement
            end
        end
    else
        X = !(isa(X, Expr) && X.head == :tuple) ? Tuple(eval_arg(caller, X)) : X #NOTE: if X is not a literally written set, then we evaluate it in the caller (e.g., I= (:x1, :y1, :z1); J= (:x2, :y2, :z2); @∀ (i,j) ∈ zip(I,J) V2.i = V.j)
        X = !isa(X, Tuple) ? Tuple(extract_tuple(X; nested=true)) : X
        x = Tuple(extract_tuple(x; nested=true))
        return quote
            $((substitute(statement, NamedTuple{x}(!isa(x_val, Tuple) ? Tuple(extract_tuple(x_val; nested=true)) : x_val); inQuoteNode=true) for x_val in X)...)
        end
    end
end


## FUNCTIONS FOR DIVERSE TASKS

function return_value(value)
    return :(return $value)
end

function return_nothing()
    return :(return)
end

function threads(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_THREADS)   return :(Base.Threads.@threads($(args...)))
    elseif (package == PKG_POLYESTER) return :(Polyester.@batch($(args...)))
    elseif isgpu(package)             return :(begin end)
    else                              @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function _firstindex(caller::Module, A::Union{Symbol, Expr}, dim::Union{Integer, Symbol, Expr}, padding::Union{Bool, Symbol, Expr}=false)
    padding = eval_arg(caller, padding)
    if (padding) return :($A.indices[$dim][1])
    else         return :(1)
    end
end

function _lastindex(caller::Module, A::Union{Symbol, Expr}, dim::Union{Integer, Symbol, Expr}, padding::Union{Bool, Symbol, Expr}=false)
    padding = eval_arg(caller, padding)
    if (padding) return :($A.indices[$dim][end])
    else         return :(size($A, $dim))
    end
end


## CPU TARGET IMPLEMENTATIONS

macro gridDim_cpu()   esc(:(ParallelStencil.ParallelKernel.Dim3($(RANGELENGTHS_VARNAMES[1]), $(RANGELENGTHS_VARNAMES[2]), $(RANGELENGTHS_VARNAMES[3])))) end

macro blockIdx_cpu()  esc(:(ParallelStencil.ParallelKernel.Dim3($(INDICES[1]) - $RANGES_VARNAME[1][1] + 1,
                                                                $(INDICES[2]) - $RANGES_VARNAME[2][1] + 1,
                                                                $(INDICES[3]) - $RANGES_VARNAME[3][1] + 1
                                                               ))) end

macro blockDim_cpu()  esc(:(ParallelStencil.ParallelKernel.Dim3(1, 1, 1))) end

macro threadIdx_cpu() esc(:(ParallelStencil.ParallelKernel.Dim3(1, 1, 1))) end

macro sync_threads_cpu() esc(:(begin end)) end

macro sharedMem_cpu(T, dims) :(MArray{Tuple{$(esc(dims))...}, $(esc(T)), length($(esc(dims))), prod($(esc(dims)))}(undef)); end # Note: A macro is used instead of a function as a creating a type stable function is not really possible (dims can take any values and they become part of the MArray type...). MArray is not escaped in order not to have to import StaticArrays in the user code.

macro sharedMem_cpu(T, dims, offset) esc(:(ParallelStencil.ParallelKernel.@sharedMem_cpu($T, $dims))) end

## CPU BACKEND: WARP-LEVEL PRIMITIVES (zero-overhead pure functions)

# The CPU backend follows a single-thread-per-block model. All warp-level
# operations therefore degenerate to constants or identity operations.
# These functions are intentionally small, @inline, allocation-free, and
# operate on isbits values only. They are called by the macro dispatchers
# for the CPU backend.

@inline warpsize_cpu()::Int = 1

@inline laneid_cpu()::Int = 1

@inline active_mask_cpu()::UInt64 = UInt64(0x1)

# Shuffle: direct, with optional width. Identity on CPU.
@inline shfl_sync_cpu(mask::Unsigned, val, lane0::Int64)
    val
end

@inline shfl_sync_cpu(mask::Unsigned, val, lane0::Int64, width::Int64)
    val
end

# Shuffle up
@inline shfl_up_sync_cpu(mask::Unsigned, val, delta::Int64)
    val
end

@inline shfl_up_sync_cpu(mask::Unsigned, val, delta::Int64, width::Int64)
    val
end

# Shuffle down
@inline shfl_down_sync_cpu(mask::Unsigned, val, delta::Int64)
    val
end

@inline shfl_down_sync_cpu(mask::Unsigned, val, delta::Int64, width::Int64)
    val
end

# Shuffle xor (butterfly)
@inline shfl_xor_sync_cpu(mask::Unsigned, val, lane_mask0::Int64)
    val
end

@inline shfl_xor_sync_cpu(mask::Unsigned, val, lane_mask0::Int64, width::Int64)
    val
end

# Vote operations
@inline vote_any_sync_cpu(mask::Unsigned, predicate::Bool)::Bool = predicate

@inline vote_all_sync_cpu(mask::Unsigned, predicate::Bool)::Bool = predicate

# Ballot returns a mask with bit 0 set iff predicate is true; CPU uses 64-bit mask.
@inline vote_ballot_sync_cpu(mask::Unsigned, predicate::Bool)::UInt64 = predicate ? UInt64(0x1) : UInt64(0x0)
