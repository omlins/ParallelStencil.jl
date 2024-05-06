#NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1).
# The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
# "Shared" memory (which will belong to the only thread in the block) will be allocated directly in the loop (i.e., for each parallel index) as local variable.

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

Call a macro analogue to `Base.@show`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@show for Threads and CUDA.@cushow for CUDA).
"""
@doc PKSHOW_DOC
macro pk_show(args...) check_initialized(__module__); esc(pk_show(__module__, args...)); end


##
const PKPRINTLN_DOC = """
    @pk_println(...)

Call a macro analogue to `Base.@println`, compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Base.@println for Threads and CUDA.@cuprintln for CUDA).
"""
@doc PKPRINTLN_DOC
macro pk_println(args...) check_initialized(__module__); esc(pk_println(__module__, args...)); end


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


## FUNCTIONS FOR INDEXING AND DIMENSIONS

function gridDim(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.gridDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.gridGroupDim($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@gridDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockIdx(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation relies on the fact that ranges are always of type UnitRange. If this changes, then this function needs to be adapted.
    if     (package == PKG_CUDA)    return :(CUDA.blockIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupIdx($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@blockIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function blockDim(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.blockDim($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workgroupDim($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@blockDim_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function threadIdx(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks.
    if     (package == PKG_CUDA)    return :(CUDA.threadIdx($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.workitemIdx($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@threadIdx_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SYNCHRONIZATION

function sync_threads(caller::Module, args...; package::Symbol=get_package(caller)) #NOTE: the CPU implementation follows the model that no threads are grouped into blocks, i.e. that each block contains only 1 thread (with thread ID 1). The parallelization happens only over the blocks. Synchronization within a block is therefore not needed (as it contains only one thread).
    if     (package == PKG_CUDA)    return :(CUDA.sync_threads($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.sync_workgroup($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@sync_threads_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION

function sharedMem(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cuDynamicSharedMem($(args...)))
    elseif (package == PKG_AMDGPU)  return :(ParallelStencil.ParallelKernel.@sharedMem_amdgpu($(args...)))
    elseif (package == PKG_THREADS) return :(ParallelStencil.ParallelKernel.@sharedMem_cpu($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

macro sharedMem_amdgpu(T, dims) esc(:(AMDGPU.@ROCDynamicLocalArray($T, $dims, false))) end

macro sharedMem_amdgpu(T, dims, offset) esc(:(ParallelStencil.ParallelKernel.@sharedMem_amdgpu($T, $dims))) end


## FUNCTIONS FOR PRINTING

function pk_show(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cushow($(args...)))
    elseif (package == PKG_AMDGPU)  @KeywordArgumentError("this functionality is not yet supported in AMDGPU.jl.")
    elseif (package == PKG_THREADS) return :(Base.@show($(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function pk_println(caller::Module, args...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    return :(CUDA.@cuprintln($(args...)))
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.@rocprintln($(args...)))
    elseif (package == PKG_THREADS) return :(Base.println($(args...)))
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
