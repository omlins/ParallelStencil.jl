const PARALLEL_DOC = """
    @parallel kernelcall

!!! note "Advanced"
        @parallel ranges kernelcall
        @parallel nblocks nthreads kernelcall
        @parallel ranges nblocks nthreads kernelcall
        @parallel (...) kwargs... kernelcall

Declare the `kernelcall` parallel. The kernel will automatically be called as required by the package for parallelization selected with [`@init_parallel_kernel`](@ref). Synchronizes at the end of the call (if a stream is given via keyword arguments, then it synchronizes only this stream).

# Arguments
- `kernelcall`: a call to a kernel that is declared parallel.
!!! note "Advanced optional arguments"
    - `ranges::Tuple{UnitRange{},UnitRange{},UnitRange{}} | Tuple{UnitRange{},UnitRange{}} | Tuple{UnitRange{}} | UnitRange{}`: the ranges of indices in each dimension for which computations must be performed.
    - `nblocks::Tuple{Integer,Integer,Integer}`: the number of blocks to be used if the package CUDA was selected with [`@init_parallel_kernel`](@ref).
    - `nthreads::Tuple{Integer,Integer,Integer}`: the number of threads to be used if the package CUDA was selected with [`@init_parallel_kernel`](@ref).
    - `kwargs...`: keyword arguments to be passed further to CUDA (ignored for Threads).

!!! note "Performance note"
    Kernel launch parameters are automatically defined with heuristics, where not defined with optional kernel arguments. For CUDA `nthreads` is whenever reasonable set to (32,8,1) and `nblocks` accordingly to ensure that enough threads are launched.

See also: [`@init_parallel_kernel`](@ref)
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(); checkargs_parallel(args...); esc(parallel(args...)); end


##
const PARALLEL_INDICES_DOC = """
    @parallel_indices indices kernel

Declare the `kernel` parallel and generate the given parallel `indices` inside the `kernel` using the package for parallelization selected with [`@init_parallel_kernel`](@ref).
"""
@doc PARALLEL_INDICES_DOC
macro parallel_indices(args...) check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(args...)); end


##
const PARALLEL_ASYNC_DOC = """
@parallel_async kernelcall

!!! note "Advanced"
        @parallel_async ranges kernelcall
        @parallel_async nblocks nthreads kernelcall
        @parallel_async ranges nblocks nthreads kernelcall
        @parallel_async (...) kwargs... kernelcall

Declare the `kernelcall` parallel as with [`@parallel`](@ref) (see [`@parallel`](@ref) for more information); deactivates however automatic synchronization at the end of the call. Use [`@synchronize`](@ref) for synchronizing.

!!! note "Performance note"
    @parallel_async falls currently back to running synchronously if the package Threads was selected with [`@init_parallel_kernel`](@ref).

See also: [`@synchronize`](@ref), [`@parallel`](@ref)
"""
@doc PARALLEL_ASYNC_DOC
macro parallel_async(args...) check_initialized(); checkargs_parallel(args...); esc(parallel_async(args...)); end


##
const SYNCHRONIZE_DOC = """
    @synchronize()

Synchronize the GPU/CPU.

See also: [`@parallel_async`](@ref)
"""
@doc SYNCHRONIZE_DOC
macro synchronize(args...) check_initialized(); esc(synchronize(args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)            check_initialized(); checkargs_parallel(args...); esc(parallel(args...; package=PKG_CUDA)); end
macro parallel_threads(args...)         check_initialized(); checkargs_parallel(args...); esc(parallel(args...; package=PKG_THREADS)); end
macro parallel_indices_cuda(args...)    check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(args...; package=PKG_CUDA)); end
macro parallel_indices_threads(args...) check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(args...; package=PKG_THREADS)); end
macro parallel_async_cuda(args...)      check_initialized(); checkargs_parallel(args...); esc(parallel_async(args...; package=PKG_CUDA)); end
macro parallel_async_threads(args...)   check_initialized(); checkargs_parallel(args...); esc(parallel_async(args...; package=PKG_THREADS)); end
macro synchronize_cuda(args...)         check_initialized(); esc(synchronize(args...; package=PKG_CUDA)); end
macro synchronize_threads(args...)      check_initialized(); esc(synchronize(args...; package=PKG_THREADS)); end


## ARGUMENT CHECKS

function checkargs_parallel(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    if !is_call(args[end]) @ArgumentError("the last argument must be a kernel call (obtained: $(args[end])).") end
    posargs, = split_parallel_args(args)
    if length(posargs) > 3 @ArgumentError("too many positional arguments.") end
    kernelcall = args[end]
    if length(extract_kernelcall_args(kernelcall)[2]) > 0 @ArgumentError("keyword arguments are not allowed in @parallel kernel calls.") end
end

function checkargs_parallel_indices(args...)
    if (length(args) != 2) @ArgumentError("wrong number of (positional) arguments.") end
    if !is_kernel(args[end]) @ArgumentError("the last argument must be a kernel definition (obtained: $(args[end])).") end
    kernel = args[end]
    if length(extract_kernel_args(kernel)[2]) > 0 @ArgumentError("keyword arguments are not allowed in the signature of @parallel_indices kernels.") end
end


## GATEWAY FUNCTIONS

parallel_async(args::Union{Symbol,Expr}...; package::Symbol=get_package()) = parallel(args...; package=package, async=true)

function parallel(args::Union{Symbol,Expr}...; package::Symbol=get_package(), async::Bool=false)
    posargs, kwargs, kernelarg = split_parallel_args(args)
    if     (package == PKG_CUDA)    parallel_call_cuda(posargs..., kernelarg, kwargs, async)
    elseif (package == PKG_THREADS) parallel_call_threads(posargs..., kernelarg, async) # Ignore keyword args as they are not for the threads case (noted in doc).
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function parallel_indices(args::Union{Symbol,Expr}...; package::Symbol=get_package())
    numbertype = get_numbertype()
    parallel_kernel(package, numbertype, args...)
end

function synchronize(args::Union{Symbol,Expr}...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    synchronize_cuda(args...)
    elseif (package == PKG_THREADS) synchronize_threads(args...)
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## @PARALLEL KERNEL FUNCTIONS

function parallel_kernel(package::Symbol, numbertype::DataType, indices::Union{Symbol,Expr}, kernel::Expr)
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices or a single index (e.g. (ix, iy, iz) or (ix, iy) or ix ).") end
    indices = extract_tuple(indices)
    body = get_body(kernel)
    body = remove_return(body)
    if (package == PKG_CUDA)
        kernel = substitute(kernel, :(Data.Array),         :(Data.DeviceArray))
        kernel = substitute(kernel, :(Data.Cell),          :(Data.DeviceCell))
        kernel = substitute(kernel, :(Data.CellArray),  :(Data.DeviceCellArray))
        kernel = substitute(kernel, :(Data.TArray),        :(Data.DeviceTArray))
        kernel = substitute(kernel, :(Data.TCell),         :(Data.DeviceTCell))
        kernel = substitute(kernel, :(Data.TCellArray), :(Data.DeviceTCellArray))
    end
    kernel = push_to_signature!(kernel, :($RANGES_VARNAME::$RANGES_TYPE))
    if     (package == PKG_CUDA)    int_type = INT_CUDA
    elseif (package == PKG_THREADS) int_type = INT_THREADS
    end
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[1])::$int_type))
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[2])::$int_type))
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[3])::$int_type))
    ranges = [:($RANGES_VARNAME[1]), :($RANGES_VARNAME[2]), :($RANGES_VARNAME[3])]
    if (package == PKG_CUDA)
        body = add_threadids(indices, ranges, body)
        body = literaltypes(numbertype, body)
    elseif (package == PKG_THREADS)
        body = add_loop(indices, ranges, body)
        body = literaltypes(numbertype, body)
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
    body = add_return(body)
    set_body!(kernel, body)
    return kernel
end


## @PARALLEL CALL FUNCTIONS

function parallel_call_cuda(ranges::Union{Symbol,Expr}, nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, kwargs::Array, async::Bool)
    ranges = :(ParallelStencil.ParallelKernel.promote_ranges($ranges))
    push!(kernelcall.args, ranges)
    push!(kernelcall.args, :($INT_CUDA(length($ranges[1]))))
    push!(kernelcall.args, :($INT_CUDA(length($ranges[2]))))
    push!(kernelcall.args, :($INT_CUDA(length($ranges[3]))))
    synccall = async ? :(begin end) : create_synccall_cuda(kwargs)
    return :( CUDA.@cuda blocks=$nblocks threads=$nthreads $(kwargs...) $kernelcall; $synccall )
end

function parallel_call_cuda(nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, kwargs::Array, async::Bool)
    maxsize = :( $nblocks .* $nthreads )
    ranges  = :(ParallelStencil.ParallelKernel.compute_ranges($maxsize))
    parallel_call_cuda(ranges, nblocks, nthreads, kernelcall, kwargs, async)
end

function parallel_call_cuda(ranges::Union{Symbol,Expr}, kernelcall::Expr, kwargs::Array, async::Bool)
    maxsize  = :(length.(ParallelStencil.ParallelKernel.promote_ranges($ranges)))
    nthreads = :( ParallelStencil.ParallelKernel.compute_nthreads($maxsize) )
    nblocks  = :( ParallelStencil.ParallelKernel.compute_nblocks($maxsize, $nthreads) )
    parallel_call_cuda(ranges, nblocks, nthreads, kernelcall, kwargs, async)
end

function parallel_call_cuda(kernelcall::Expr, kwargs::Array, async::Bool)
    ranges = :( ParallelStencil.ParallelKernel.get_ranges($(kernelcall.args[2:end]...)) )
    parallel_call_cuda(ranges, kernelcall, kwargs, async)
end


function parallel_call_threads(ranges::Union{Symbol,Expr}, kernelcall::Expr, async::Bool)
    ranges = :(ParallelStencil.ParallelKernel.promote_ranges($ranges))
    push!(kernelcall.args, ranges)
    push!(kernelcall.args, :($INT_THREADS(length($ranges[1]))))
    push!(kernelcall.args, :($INT_THREADS(length($ranges[2]))))
    push!(kernelcall.args, :($INT_THREADS(length($ranges[3]))))
    if async
        return kernelcall # NOTE: This cannot be used currently as there is no obvious solution how to sync in this case.
    else
        return kernelcall
    end
end

function parallel_call_threads(ranges::Union{Symbol,Expr}, nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, async::Bool)
    parallel_call_threads(ranges, kernelcall, async)
end

function parallel_call_threads(nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, async::Bool)
    maxsize = :( $nblocks .* $nthreads )
    ranges  = :(ParallelStencil.ParallelKernel.compute_ranges($maxsize))
    parallel_call_threads(ranges, kernelcall, async)
end

function parallel_call_threads(kernelcall::Expr, async::Bool)
    ranges = :( ParallelStencil.ParallelKernel.get_ranges($(kernelcall.args[2:end]...)) )
    parallel_call_threads(ranges, kernelcall, async)
end


## @SYNCHRONIZE FUNCTIONS

synchronize_cuda(args::Union{Symbol,Expr}...) = :(CUDA.synchronize($(args...)))
synchronize_threads(args::Union{Symbol,Expr}...) = :(begin end)


## MACROS AND FUNCTIONS TO MODIFY THE TYPE OF LITERALS

macro literaltypes(type, expr)
    type_val = eval_arg(__module__, type)
    check_literaltype(type_val)
    esc(literaltypes(type_val, expr))
end

macro literaltypes(type1, type2, expr)
    type1_val = eval_arg(__module__, type1)
    type2_val = eval_arg(__module__, type2)
    check_literaltype(type1_val)
    check_literaltype(type2_val)
    esc(literaltypes(type1_val, type2_val, expr))
end

function literaltypes(type::DataType, expr::Expr)
    args = expr.args
    head = expr.head
    for i=1:length(args)
        if type <: Integer && typeof(args[i]) <: Integer && head == :call && args[1] in OPERATORS # NOTE: only integers in operator calls are modified (not in other function calls as e.g. size).
            literal = type(args[i])
            args[i] = :($literal)
        elseif type <: AbstractFloat && typeof(args[i]) <: AbstractFloat
            literal = type(args[i])
            args[i] = :($literal)
        elseif type <: Complex{<:AbstractFloat} && typeof(args[i]) <: AbstractFloat # NOTE: complex float numbers decompose into two normal floats with both the same type
            literal = type.types[1](args[i])
            args[i] = :($literal)
        elseif typeof(args[i]) == Expr
            args[i] = literaltypes(type, args[i])
        end
    end
    return expr
end

function literaltypes(type1::DataType, type2::DataType, expr::Expr)
    if !((type1 <: Integer && type2 <: AbstractFloat) || (type1 <: AbstractFloat && type2 <: Integer))
        @ModuleInternalError("incoherent type arguments: one must be a subtype of Integer and the other a subtype of AbstractFloat.")
    end
    return literaltypes(type2, literaltypes(type1,expr))
end


## FUNCTIONS TO ADD THREAD-IDS / LOOPS IN KERNELS

function add_threadids(indices::Array, ranges::Array, block::Expr)
    if !(length(ranges)==3) @ModuleInternalError("ranges must be an Array or Tuple of size 3.") end # E.g. (5:28,5:28,1:1) in 2D. Note that for simplicity for the users and developers, 1D and 2D problems are always expressed like 3D problems...
    check_thread_bounds = true
    if haskey(ENV, "PS_THREAD_BOUND_CHECK") check_thread_bounds = (parse(Int64, ENV["PS_THREAD_BOUND_CHECK"]) > 0); end # PS_THREAD_BOUND_CHECK=0 enables to deactivate the check whether each thread is in bounds of the ranges array in order to reach maximal performance. If deactivated and any thread is out-of-bound it will cause normally a segmentation fault. To ensure that all threads are in bounds, the thread block must be of the same size as the ranges passed to the @parallel or @parallel_indices function.
    thread_bounds_check = :(begin end)
    rangelength_x, rangelength_y, rangelength_z = RANGELENGTHS_VARNAMES
    tx, ty, tz = THREADIDS_VARNAMES
    ndims = length(indices)
    if ndims == 1
        ix, = indices
        range_x, = ranges
        if check_thread_bounds
            thread_bounds_check = quote
                if ($tx > $rangelength_x) return; end
            end
        end
        quote
            $tx = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x;  # thread ID, dimension x
            $thread_bounds_check
            $ix = $range_x[$tx]                                                    # index, dimension x
            $block
        end
    elseif ndims == 2
        ix, iy = indices
        range_x, range_y = ranges
        if check_thread_bounds
            thread_bounds_check = quote
                if ($tx > $rangelength_x) return; end
                if ($ty > $rangelength_y) return; end
            end
        end
        quote
            $tx = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x;  # thread ID, dimension x
            $ty = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y;  # thread ID, dimension y
            $thread_bounds_check
            $ix = $range_x[$tx]                                                    # index, dimension x
            $iy = $range_y[$ty]                                                    # index, dimension y
            $block
        end
    elseif ndims == 3
        ix, iy, iz = indices
        range_x, range_y, range_z = ranges
        if check_thread_bounds
            thread_bounds_check = quote
                if ($tx > $rangelength_x) return; end
                if ($ty > $rangelength_y) return; end
                if ($tz > $rangelength_z) return; end
            end
        end
        quote
            $tx = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x;  # thread ID, dimension x
            $ty = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y;  # thread ID, dimension y
            $tz = (CUDA.blockIdx().z-1) * CUDA.blockDim().z + CUDA.threadIdx().z;  # thread ID, dimension z
            $thread_bounds_check
            $ix = $range_x[$tx]                                                    # index, dimension x
            $iy = $range_y[$ty]                                                    # index, dimension y
            $iz = $range_z[$tz]                                                    # index, dimension z
            $block
        end
    end
end

function add_loop(indices::Array, ranges::Array, block::Expr)
    if !(length(ranges)==3) @ModuleInternalError("ranges must be an Array or Tuple of size 3") end # E.g. (5:28,5:28,1:1) in 2D.
    range_x, range_y, range_z = ranges
    ndims = length(indices)
    if ndims == 1
        ix, = indices
        ix_ps, iy_ps, iz_ps = INDICES
        ix_ps_assignment = (ix_ps!=ix) ? :($ix_ps = $ix) : :(begin end)  # Note: this assignement is only required in parallel_indices kernels, where the user chooses the index names freely. Assingning it to the know index name allows to use it in application independent macros (e.g. blockIdx).
        iy_ps_assignment = :($iy_ps = $range_y[1])
        iz_ps_assignment = :($iz_ps = $range_z[1])
        quote
            $iz_ps_assignment
            $iy_ps_assignment
            Base.Threads.@threads for $ix in $range_x
                $ix_ps_assignment
                $block
            end
        end
    elseif ndims == 2
        ix, iy = indices
        ix_ps, iy_ps, iz_ps = INDICES
        ix_ps_assignment = (ix_ps!=ix) ? :($ix_ps = $ix) : :(begin end)  # Note: this assignement is only required in parallel_indices kernels, where the user chooses the index names freely.
        iy_ps_assignment = (iy_ps!=iy) ? :($iy_ps = $iy) : :(begin end)  # ...
        iz_ps_assignment = :($iz_ps = $range_z[1])
        quote
            $iz_ps_assignment
            Base.Threads.@threads for $iy in $range_y
                $iy_ps_assignment
                for $ix in $range_x
                    $ix_ps_assignment
                    $block
                end
            end
        end
    elseif ndims == 3
        ix, iy, iz = indices
        ix_ps, iy_ps, iz_ps = INDICES
        ix_ps_assignment = (ix_ps!=ix) ? :($ix_ps = $ix) : :(begin end)  # Note: this assignement is only required in parallel_indices kernels, where the user chooses the index names freely.
        iy_ps_assignment = (iy_ps!=iy) ? :($iy_ps = $iy) : :(begin end)  # ...
        iz_ps_assignment = (iz_ps!=iz) ? :($iz_ps = $iz) : :(begin end)  # ...
        quote
            Base.Threads.@threads for $iz in $range_z
                $iz_ps_assignment
                for $iy in $range_y
                    $iy_ps_assignment
                    for $ix in $range_x
                        $ix_ps_assignment
                        $block
                    end
                end
            end
        end
    end
end


## FUNCTIONS TO GET RANGES, PROMOTE THEM TO 3D, AND COMPUTE NTHREADS, NBLOCKS AND RANGES

promote_ranges(ranges::RANGES_TYPE_1D)          = (ranges,    1:1, 1:1)
promote_ranges(ranges::RANGES_TYPE_1D_TUPLE)    = (ranges..., 1:1, 1:1)
promote_ranges(ranges::RANGES_TYPE_2D)          = (ranges..., 1:1)
promote_ranges(ranges::RANGES_TYPE)             = ranges
promote_ranges(ranges)                          = @ModuleInternalError("ranges must be a Tuple of UnitRange of size 1, 2 or 3 (obtained: $ranges; its type is: $(typeof(ranges))).")

promote_maxsize(maxsize::MAXSIZE_TYPE_1D)       = (maxsize,    1, 1)
promote_maxsize(maxsize::MAXSIZE_TYPE_1D_TUPLE) = (maxsize..., 1, 1)
promote_maxsize(maxsize::MAXSIZE_TYPE_2D)       = (maxsize..., 1)
promote_maxsize(maxsize::MAXSIZE_TYPE)          = maxsize
promote_maxsize(maxsize)                        = @ModuleInternalError("maxsize must be a Tuple of Integer of size 1, 2 or 3 (obtained: $maxsize; its type is: $(typeof(maxsize))).")

maxsize(A::T) where T<:AbstractArray = (size(A,1),size(A,2),size(A,3))          # NOTE: using size(A,dim) three times instead of size(A) ensures to have a tuple of length 3.
maxsize(a::T) where T<:Number        = (1, 1, 1)
maxsize(x)                           = @ArgumentError("automatic detection of ranges not possible in @parallel <kernelcall>: some kernel arguments are neither arrays nor scalars. Specify ranges or nthreads and nblocks manually.")
maxsize(x, args...)                  = merge(maxsize(x), maxsize(args...))      # NOTE: maxsize is implemented as a recursive function, which results in optimal code; otherwise, the function is not performance-negligable for small problems.
merge(a::Tuple, b::Tuple)            = max.(a,b)

function get_ranges(args...)
    maxsizes = maxsize(args...)
    return (1:maxsizes[1], 1:maxsizes[2], 1:maxsizes[3])
end

function compute_ranges(maxsize)
    maxsize = promote_maxsize(maxsize)
    return (1:maxsize[1], 1:maxsize[2], 1:maxsize[3])
end

function compute_nthreads(maxsize; nthreads_max=NTHREADS_MAX, flatdim=0) # This is a heuristic, which results in (32,8,1) threads, except if maxsize[1] < 32 or maxsize[2] < 8.
    maxsize = promote_maxsize(maxsize)
    nthreads_x = min(32,                                             (flatdim==1) ? 1 : maxsize[1])
    nthreads_y = min(ceil(Int,nthreads_max/nthreads_x),              (flatdim==2) ? 1 : maxsize[2])
    nthreads_z = min(ceil(Int,nthreads_max/(nthreads_x*nthreads_y)), (flatdim==3) ? 1 : maxsize[3])
    return (nthreads_x, nthreads_y , nthreads_z)
end

function compute_nblocks(maxsize, nthreads)
    maxsize = promote_maxsize(maxsize)
    if !(isa(nthreads, Union{AbstractArray,Tuple}) && length(nthreads)==3) @ArgumentError("nthreads must be an Array or Tuple of size 3 (obtained: $nthreads; its type is: $(typeof(nthreads))).") end
    return ceil.(Int, maxsize./nthreads)
end


## FUNCTIONS TO CREATE SYNCHRONIZATION CALLS

function create_synccall_cuda(kwargs::Array)
    kwarg_stream = [x for x in kwargs if x.args[1]==:stream]
    if length(kwarg_stream) == 0
        synchronize_cuda()
    elseif length(kwarg_stream) == 1
        stream = kwarg_stream[1].args[2]
        synchronize_cuda(stream)
    else
        @KeywordArgumentError("there can only be one keyword argument stream in a @parallel <kernelcall>.")
    end
end

function create_synccall_threads(kwargs::Array)
    synchronize_threads()
end
