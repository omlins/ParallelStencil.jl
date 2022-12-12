#NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1).
# The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
# "Shared" memory (which will belong to the only thread in the block) will be allocated directly in the loop (i.e., for each parallel index) as local variable.

##
const GRIDDIM_DOC = """
    @gridDim()

Return the grid size (or "dimension") in x, y and z dimension. The grid size in a specific dimension is commonly retrieved directly as in this example in x dimension: `@gridDim().x`.
"""
@doc GRIDDIM_DOC
macro gridDim(args...) check_initialized(); checknoargs(args...); esc(gridDim(args...)); end


##
const BLOCKIDX_DOC = """
    @blockIdx()

Return the block ID in x, y and z dimension within the grid. The block ID in a specific dimension is commonly retrieved directly as in this example in x dimension: `@blockIdx().x`.
"""
@doc BLOCKIDX_DOC
macro blockIdx(args...) check_initialized(); checknoargs(args...); esc(blockIdx(args...)); end


##
const BLOCKDIM_DOC = """
    @blockDim()

Return the block size (or "dimension") in x, y and z dimension. The block size in a specific dimension is commonly retrieved directly as in this example in x dimension: `@blockDim().x`.
"""
@doc BLOCKDIM_DOC
macro blockDim(args...) check_initialized(); checknoargs(args...); esc(blockDim(args...)); end


##
const THREADIDX_DOC = """
    @threadIdx()

Return the thread ID in x, y and z dimension within the block. The thread ID in a specific dimension is commonly retrieved directly as in this example in x dimension: `@threadIdx().x`.
"""
@doc THREADIDX_DOC
macro threadIdx(args...) check_initialized(); checknoargs(args...); esc(threadIdx(args...)); end


##
const SYNCTHREADS_DOC = """
    @sync_threads()

Synchronize the threads of the block: wait until all threads in the block have reached this point and all global and shared memory accesses made by these threads prior to the `sync_threads()` call are visible to all threads in the block.
"""
@doc SYNCTHREADS_DOC
macro sync_threads(args...) check_initialized(); checknoargs(args...); esc(sync_threads(args...)); end


##
# NOTE: the optional offset parameter of cuDynamicSharedMem is currently not exposed.
const SHAREDMEM_DOC = """
    @sharedMem(T, dims)

Create an array that is *shared* between the threads of a block (i.e. accessible only by the threads of a same block), with element type `T` and size specified by `dims`.

!!! note "Note"
    The amount of shared memory needs to specified when launching the kernel (keyword argument `shmem`).
"""
@doc SHAREDMEM_DOC
macro sharedMem(args...) check_initialized(); checkargs_sharedMem(args...); esc(sharedMem(args...)); end


##
const PKSHOW_DOC = """
    @pk_show(...)

Call a macro analogue to `Base.@show`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@show for Threads and CUDA.@cushow for CUDA).
"""
@doc PKSHOW_DOC
macro pk_show(args...) check_initialized(); esc(pk_show(args...)); end


##
const PKPRINTLN_DOC = """
    @pk_println(...)

Call a macro analogue to `Base.@println`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@println for Threads and CUDA.@cuprintln for CUDA).
"""
@doc PKPRINTLN_DOC
macro pk_println(args...) check_initialized(); esc(pk_println(args...)); end


## ARGUMENT CHECKS

function checknoargs(args...)
    if (length(args) != 0) @ArgumentError("no arguments allowed.") end
end

function checkargs_sharedMem(args...)
    if (length(args) != 2) @ArgumentError("wrong number of arguments.") end
end


## FUNCTIONS FOR INDEXING AND DIMENSIONS

function gridDim(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(CUDA.gridDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.gridDimWG($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@gridDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockIdx(args...; package::Symbol=get_package()) #NOTE: the CPU implementation relies on the fact that ranges are always of type UnitRange. If this changes, then this function needs to be adapted.
    if     (package == PKG_CUDA)    return :(CUDA.blockIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupIdx($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@blockIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockDim(args...; package::Symbol=get_package()) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.blockDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupDim($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@blockDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function threadIdx(args...; package::Symbol=get_package()) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.threadIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workitemIdx($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@threadIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SYNCHRONIZATION

function sync_threads(args...; package::Symbol=get_package()) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
    if     (package == PKG_CUDA)    return :(CUDA.sync_threads($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.sync_workgroup($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@sync_threads_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION

function sharedMem(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(CUDA.@cuDynamicSharedMem($(args...)))
    elseif (package == PKG_AMDGPU)  @KeywordArgumentError("not yet supported for AMDGPU.")
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@sharedMem_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR PRINTING

function pk_show(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(CUDA.@cushow($(args...)))
    elseif (package == PKG_AMDGPU)  @KeywordArgumentError("this functionality is not yet supported in AMDGPU.jl.")
    elseif (package == PKG_THREADS) return :(Base.@show($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function pk_println(args...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    return :(CUDA.@cuprintln($(args...)))
    elseif (package == PKG_AMDGPU)  @KeywordArgumentError("this functionality is not yet supported in AMDGPU.jl.")
    elseif (package == PKG_THREADS) return :(Base.@println($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
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
