const PARALLEL_DOC = """
    @parallel kernelcall
    @parallel ∇=... kernelcall

!!! note "Advanced"
        @parallel ranges kernelcall
        @parallel nblocks nthreads kernelcall
        @parallel ranges nblocks nthreads kernelcall
        @parallel (...) configcall=... backendkwargs... kernelcall
        @parallel ∇=... ad_mode=... ad_annotations=... (...) backendkwargs... kernelcall

Declare the `kernelcall` parallel. The kernel will automatically be called as required by the package for parallelization selected with [`@init_parallel_kernel`](@ref). Synchronizes at the end of the call (if a stream is given via keyword arguments, then it synchronizes only this stream). The keyword argument `∇` triggers a parallel call to the gradient kernel instead of the kernel itself. The automatic differentiation is performed with the package Enzyme.jl (refer to the corresponding documentation for Enzyme-specific terms used below).

# Arguments
- `kernelcall`: a call to a kernel that is declared parallel.
!!! note "Advanced optional arguments"
    - `ranges::Tuple{UnitRange{},UnitRange{},UnitRange{}} | Tuple{UnitRange{},UnitRange{}} | Tuple{UnitRange{}} | UnitRange{}`: the ranges of indices in each dimension for which computations must be performed.
    - `nblocks::Tuple{Integer,Integer,Integer}`: the number of blocks to be used if the package CUDA or AMDGPU was selected with [`@init_parallel_kernel`](@ref).
    - `nthreads::Tuple{Integer,Integer,Integer}`: the number of threads to be used if the package CUDA or AMDGPU was selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
!!! note "Advanced"
    - `∇`: the variable(s) with respect to which the kernel is to be differentiated automatically and a duplicate for each variable to store the result in, separated by `->`, e.g., `∇=(A->Ā, B->B̄)`. Setting this keyword triggers a parallel call to the gradient kernel instead of the kernel itself. The duplicate variables are by default passed to Enzyme with the annotation `DuplicatedNoNeed`, e.g., `DuplicatedNoNeed(A, Ā)`. Use the keyword argument `ad_annotations` to modify this behavior.
    - `ad_mode=Enzyme.Reverse`: the automatic differentiation mode (see the documentation of Enzyme.jl for more information).
    - `ad_annotations=()`: Enzyme variable annotations for automatic differentiation in the format `(<keyword>=<variable(s)>, <keyword>=<variable(s)>, ...)`, where `<variable(s)>` can be a single variable or a tuple of variables (e.g., `ad_annotations=(Duplicated=B, Active=(a,b))`). Currently supported annotations are: $(keys(AD_SUPPORTED_ANNOTATIONS)).
    - `configcall=kernelcall`: a call to a kernel that is declared parallel, which is used for determining the kernel launch parameters. This keyword is useful, e.g., for generic automatic differentiation using the low-level submodule [`AD`](@ref).
    - `backendkwargs...`: keyword arguments to be passed further to CUDA or AMDGPU (ignored for Threads).

!!! note "Performance note"
    Kernel launch parameters are automatically defined with heuristics, where not defined with optional kernel arguments. For CUDA and AMDGPU, `nthreads` is typically set to (32,8,1) and `nblocks` accordingly to ensure that enough threads are launched.

See also: [`@init_parallel_kernel`](@ref)
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__module__, args...)); end


##
const PARALLEL_INDICES_DOC = """
    @parallel_indices indices kernel
    @parallel_indices indices inbounds=... kernel

# Keyword arguments
- `inbounds::Bool`: whether to apply `@inbounds` to the kernel. The default is `false` or as set with the `inbounds` keyword argument of [`@init_parallel_kernel`](@ref).

Declare the `kernel` parallel and generate the given parallel `indices` inside the `kernel` using the package for parallelization selected with [`@init_parallel_kernel`](@ref).
"""
@doc PARALLEL_INDICES_DOC
macro parallel_indices(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...)); end


##
const PARALLEL_ASYNC_DOC = """
@parallel_async kernelcall
@parallel_async ∇=... kernelcall

!!! note "Advanced"
        @parallel_async ranges kernelcall
        @parallel_async nblocks nthreads kernelcall
        @parallel_async ranges nblocks nthreads kernelcall
        @parallel_async (...) configcall=... backendkwargs... kernelcall
        @parallel_async ∇=... ad_mode=... ad_annotations=... (...) backendkwargs... kernelcall

Declare the `kernelcall` parallel as with [`@parallel`](@ref) (see [`@parallel`](@ref) for more information); deactivates however automatic synchronization at the end of the call. Use [`@synchronize`](@ref) for synchronizing.

!!! note "Performance note"
    @parallel_async falls currently back to running synchronously if the package Threads was selected with [`@init_parallel_kernel`](@ref).

See also: [`@synchronize`](@ref), [`@parallel`](@ref)
"""
@doc PARALLEL_ASYNC_DOC
macro parallel_async(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__module__, args...)); end


##
const SYNCHRONIZE_DOC = """
    @synchronize()

Synchronize the GPU/CPU.

See also: [`@parallel_async`](@ref)
"""
@doc SYNCHRONIZE_DOC
macro synchronize(args...) check_initialized(__module__); esc(synchronize(__module__, args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)            check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_CUDA)); end
macro parallel_amdgpu(args...)          check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_AMDGPU)); end
macro parallel_threads(args...)         check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_THREADS)); end
macro parallel_indices_cuda(args...)    check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...; package=PKG_CUDA)); end
macro parallel_indices_amdgpu(args...)  check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...; package=PKG_AMDGPU)); end
macro parallel_indices_threads(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...; package=PKG_THREADS)); end
macro parallel_async_cuda(args...)      check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__module__, args...; package=PKG_CUDA)); end
macro parallel_async_amdgpu(args...)    check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__module__, args...; package=PKG_AMDGPU)); end
macro parallel_async_threads(args...)   check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__module__, args...; package=PKG_THREADS)); end
macro synchronize_cuda(args...)         check_initialized(__module__); esc(synchronize(__module__, args...; package=PKG_CUDA)); end
macro synchronize_amdgpu(args...)       check_initialized(__module__); esc(synchronize(__module__, args...; package=PKG_AMDGPU)); end
macro synchronize_threads(args...)      check_initialized(__module__); esc(synchronize(__module__, args...; package=PKG_THREADS)); end


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
    posargs, = split_args(args)
    if (length(posargs) != 2) @ArgumentError("wrong number of positional arguments.") end
    if !is_kernel(args[end]) @ArgumentError("the last argument must be a kernel definition (obtained: $(args[end])).") end
    kernel = args[end]
    if length(extract_kernel_args(kernel)[2]) > 0 @ArgumentError("keyword arguments are not allowed in the signature of @parallel_indices kernels.") end
end


## GATEWAY FUNCTIONS

parallel_async(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller)) = parallel(caller, args...; package=package, async=true)

function parallel(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller), async::Bool=false)
    posargs, kwargs_expr, kernelarg = split_parallel_args(args)
    kwargs, backend_kwargs_expr = extract_kwargs(caller, kwargs_expr, (:stream, :shmem, :launch, :configcall, :∇, :ad_mode, :ad_annotations), "@parallel <kernelcall>", true; eval_args=(:launch,))
    launch          = haskey(kwargs, :launch) ? kwargs.launch : true
    configcall      = haskey(kwargs, :configcall) ? kwargs.configcall : kernelarg
    is_ad_highlevel = haskey(kwargs, :∇)
    if !is_ad_highlevel && (haskey(kwargs, :ad_mode) || haskey(kwargs, :ad_annotations)) @IncoherentArgumentError("incoherent arguments `ad_mode`/`ad_annotations` in @parallel call: AD keywords are only valid if automatic differentiation is triggered with the keyword argument `∇`.") end
    if is_ad_highlevel
        parallel_call_ad(caller, kernelarg, backend_kwargs_expr, async, package, posargs, kwargs)
    else
        if     isgpu(package)           parallel_call_gpu(posargs..., kernelarg, backend_kwargs_expr, async, package; kwargs...)
        elseif (package == PKG_THREADS) parallel_call_threads(posargs..., kernelarg, async; launch=launch, configcall=configcall) # Ignore keyword args as they are not for the threads case (noted in doc).
        else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
        end
    end
end

function parallel_indices(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller))
    numbertype = get_numbertype(caller)
    posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
    kwargs, backend_kwargs_expr = extract_kwargs(caller, kwargs_expr, (:inbounds,), "@parallel_indices <indices> <kernel>", true; eval_args=(:inbounds,))
    inbounds = haskey(kwargs, :inbounds) ? kwargs.inbounds : get_inbounds(caller)
    parallel_kernel(caller, package, numbertype, inbounds, posargs..., kernelarg)
end

function synchronize(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    synchronize_cuda(args...)
    elseif (package == PKG_AMDGPU)  synchronize_amdgpu(args...)
    elseif (package == PKG_THREADS) synchronize_threads(args...)
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end


## @PARALLEL KERNEL FUNCTIONS

function parallel_kernel(caller::Module, package::Symbol, numbertype::DataType, inbounds::Bool, indices::Union{Symbol,Expr}, kernel::Expr)
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices or a single index (e.g. (ix, iy, iz) or (ix, iy) or ix ).") end
    indices = extract_tuple(indices)
    body = get_body(kernel)
    body = remove_return(body)
    use_aliases = !all(indices .== INDICES[1:length(indices)])
    if use_aliases # NOTE: we treat explicit parallel indices as aliases to the statically retrievable indices INDICES.
        indices_aliases = indices
        indices = [INDICES[1:length(indices)]...]
        for i=1:length(indices_aliases)
            body = substitute(body, indices_aliases[i], indices[i])
        end
    end
    if isgpu(package) kernel = insert_device_types(kernel) end
    kernel = push_to_signature!(kernel, :($RANGES_VARNAME::$RANGES_TYPE))
    if     (package == PKG_CUDA)    int_type = INT_CUDA
    elseif (package == PKG_AMDGPU)  int_type = INT_AMDGPU
    elseif (package == PKG_THREADS) int_type = INT_THREADS
    end
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[1])::$int_type))
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[2])::$int_type))
    kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[3])::$int_type))
    ranges = [:($RANGES_VARNAME[1]), :($RANGES_VARNAME[2]), :($RANGES_VARNAME[3])]
    if isgpu(package)
        body = add_threadids(indices, ranges, body)        
        body = (numbertype != NUMBERTYPE_NONE) ? literaltypes(numbertype, body) : body
        body = literaltypes(int_type, body) # TODO: the size function always returns a 64 bit integer; the following is not performance efficient: body = cast(body, :size, int_type)
    elseif (package == PKG_THREADS)
        body = add_loop(indices, ranges, body)
        body = (numbertype != NUMBERTYPE_NONE) ? literaltypes(numbertype, body) : body
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
    if use_aliases
        body = macroexpand(caller, body)
        for i=1:length(indices_aliases)
            body = substitute(body, indices_aliases[i], indices[i])
        end
    end
    if (inbounds) body = add_inbounds(body) end
    body = add_return(body)
    set_body!(kernel, body)
    # @show QuoteNode(simplify_varnames!(remove_linenumbernodes!(deepcopy(kernel))))
    return kernel
end


## @PARALLEL CALL FUNCTIONS

function parallel_call_ad(caller::Module, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, package::Symbol, posargs, kwargs)
    ad_mode              = haskey(kwargs, :ad_mode) ? kwargs.ad_mode : AD_MODE_DEFAULT
    ad_annotations_expr  = haskey(kwargs, :ad_annotations) ? extract_tuple(kwargs.ad_annotations; nested=true) : []
    ad_vars_expr         = extract_tuple(kwargs.∇; nested=true)
    ~, ~, ~, ad_vars     = extract_kwargs(caller, ad_vars_expr, (), "", true; separator=:->, keyword_type=Any)
    ~, ~, ad_annotations = extract_kwargs(caller, ad_annotations_expr, (), "", true)
    ad_vars              = Dict(key => unblock(val) for (key,val) in ad_vars)
    ad_annotations       = map(x->extract_tuple(x), ad_annotations)
    f_name               = extract_kernelcall_name(kernelcall)
    f_posargs, ~         = extract_kernelcall_args(kernelcall)
    ad_annotations_byvar = Dict(a => [] for a in f_posargs)
    for (keyword, vars) in zip(keys(ad_annotations), values(ad_annotations))
        if (keyword ∉ keys(AD_SUPPORTED_ANNOTATIONS)) @KeywordArgumentError("annotation $keyword is not (yet) supported with high-level syntax; use the generic syntax calling directly `autodiff_deferred!`.") end
        for var in vars
            if (ad_annotations_byvar[var] != []) @KeywordArgumentError("variable $var has more than one annotation. Nested annotations are not (yet) supported with high-level syntax; use the generic syntax calling directly `autodiff_deferred!`.") end
            push!(ad_annotations_byvar[var], AD_SUPPORTED_ANNOTATIONS[keyword])
        end
    end
    for var in keys(ad_vars)
        if (var ∉ f_posargs) @KeywordArgumentError("variable $var in $(kwargs.∇) is not a positional argument of the kernel call $kernelcall.") end
        if ad_annotations_byvar[var] == []
            push!(ad_annotations_byvar[var], AD_DUPLICATE_DEFAULT)
        end
    end
    for var in f_posargs
        if ad_annotations_byvar[var] == []
            push!(ad_annotations_byvar[var], AD_ANNOTATION_DEFAULT)
        end
    end
    annotated_args        = (:($(ad_annotations_byvar[var][1])($((var ∈ keys(ad_vars) ? (var, ad_vars[var]) : (var,))...))) for var in f_posargs)
    ad_call               = :(ParallelStencil.ParallelKernel.AD.autodiff_deferred!($ad_mode, $f_name, $(annotated_args...)))
    kwargs_remaining      = filter(x->!(x in (:∇, :ad_mode, :ad_annotations)), keys(kwargs))
    kwargs_remaining_expr = [:($key=$val) for (key,val) in kwargs_remaining]
    if (async) return :( @parallel_async $(posargs...) $(backend_kwargs_expr...) $(kwargs_remaining_expr...) configcall=$kernelcall $ad_call ) #TODO: the package needs to be passed further here later.
    else       return :( @parallel       $(posargs...) $(backend_kwargs_expr...) $(kwargs_remaining_expr...) configcall=$kernelcall $ad_call ) #...
    end
end

function parallel_call_gpu(ranges::Union{Symbol,Expr}, nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, package::Symbol; stream::Union{Symbol,Expr}=default_stream(package), shmem::Union{Symbol,Expr,Nothing}=nothing, launch::Bool=true, configcall::Expr=kernelcall)
    ranges = :(ParallelStencil.ParallelKernel.promote_ranges($ranges))
    if     (package == PKG_CUDA)   int_type = INT_CUDA
    elseif (package == PKG_AMDGPU) int_type = INT_AMDGPU
    end
    push!(kernelcall.args, ranges) #TODO: to enable indexing with other then Int64 something like the following but probably better in a function will also be necessary: push!(kernelcall.args, :(convert(Tuple{UnitRange{$int_type},UnitRange{$int_type},UnitRange{$int_type}}, $ranges)))
    push!(kernelcall.args, :($int_type(length($ranges[1]))))
    push!(kernelcall.args, :($int_type(length($ranges[2]))))
    push!(kernelcall.args, :($int_type(length($ranges[3]))))
    return create_gpu_call(package, nblocks, nthreads, kernelcall, backend_kwargs_expr, async, stream, shmem, launch)
end

function parallel_call_gpu(nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, package::Symbol; stream::Union{Symbol,Expr}=default_stream(package), shmem::Union{Symbol,Expr,Nothing}=nothing, launch::Bool=true, configcall::Expr=kernelcall)
    maxsize = :( $nblocks .* $nthreads )
    ranges  = :(ParallelStencil.ParallelKernel.compute_ranges($maxsize))
    parallel_call_gpu(ranges, nblocks, nthreads, kernelcall, backend_kwargs_expr, async, package; stream=stream, shmem=shmem, launch=launch)
end

function parallel_call_gpu(ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, package::Symbol; stream::Union{Symbol,Expr}=default_stream(package), shmem::Union{Symbol,Expr,Nothing}=nothing, launch::Bool=true, configcall::Expr=kernelcall)
    maxsize  = :(length.(ParallelStencil.ParallelKernel.promote_ranges($ranges)))
    nthreads = :( ParallelStencil.ParallelKernel.compute_nthreads($maxsize) )
    nblocks  = :( ParallelStencil.ParallelKernel.compute_nblocks($maxsize, $nthreads) )
    parallel_call_gpu(ranges, nblocks, nthreads, kernelcall, backend_kwargs_expr, async, package; stream=stream, shmem=shmem, launch=launch)
end

function parallel_call_gpu(kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, package::Symbol; stream::Union{Symbol,Expr}=default_stream(package), shmem::Union{Symbol,Expr,Nothing}=nothing, launch::Bool=true, configcall::Expr=kernelcall)
    ranges = :( ParallelStencil.ParallelKernel.get_ranges($(configcall.args[2:end]...)) )
    parallel_call_gpu(ranges, kernelcall, backend_kwargs_expr, async, package; stream=stream, shmem=shmem, launch=launch)
end


function parallel_call_threads(ranges::Union{Symbol,Expr}, kernelcall::Expr, async::Bool; launch::Bool=true, configcall::Expr=kernelcall)
    ranges = :(ParallelStencil.ParallelKernel.promote_ranges($ranges))
    push!(kernelcall.args, ranges) #TODO: to enable indexing with other then Int64 something like the following but probably better in a function will also be necessary: push!(kernelcall.args, :(convert(Tuple{UnitRange{$INT_THREADS},UnitRange{$INT_THREADS},UnitRange{$INT_THREADS}}, $ranges)))
    push!(kernelcall.args, :($INT_THREADS(length($ranges[1]))))
    push!(kernelcall.args, :($INT_THREADS(length($ranges[2]))))
    push!(kernelcall.args, :($INT_THREADS(length($ranges[3]))))
    if launch
        if async
            return kernelcall # NOTE: This cannot be used currently as there is no obvious solution how to sync in this case.
        else
            return kernelcall
        end
    else
        return :(@which $kernelcall)
    end
end

function parallel_call_threads(ranges::Union{Symbol,Expr}, nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, async::Bool; launch::Bool=true, configcall::Expr=kernelcall)
    parallel_call_threads(ranges, kernelcall, async; launch=launch)
end

function parallel_call_threads(nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, async::Bool; launch::Bool=true, configcall::Expr=kernelcall)
    maxsize = :( $nblocks .* $nthreads )
    ranges  = :(ParallelStencil.ParallelKernel.compute_ranges($maxsize))
    parallel_call_threads(ranges, kernelcall, async; launch=launch)
end

function parallel_call_threads(kernelcall::Expr, async::Bool; launch::Bool=true, configcall::Expr=kernelcall)
    ranges = :( ParallelStencil.ParallelKernel.get_ranges($(configcall.args[2:end]...)) )
    parallel_call_threads(ranges, kernelcall, async; launch=launch)
end


## @SYNCHRONIZE FUNCTIONS

synchronize_cuda(args::Union{Symbol,Expr}...) = :(CUDA.synchronize($(args...)))
synchronize_amdgpu(args::Union{Symbol,Expr}...) = :(AMDGPU.synchronize($(args...)))
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
        if type <: Integer && typeof(args[i]) <: Integer && (head == :comparison || (head == :call && args[1] in INDEXING_OPERATORS)) # NOTE: only integers in comparisons or operator calls are modified (not in other function calls as e.g. size).
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
            $tx = (ParallelStencil.ParallelKernel.@blockIdx().x-1) * ParallelStencil.ParallelKernel.@blockDim().x + ParallelStencil.ParallelKernel.@threadIdx().x;  # thread ID, dimension x
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
            $tx = (ParallelStencil.ParallelKernel.@blockIdx().x-1) * ParallelStencil.ParallelKernel.@blockDim().x + ParallelStencil.ParallelKernel.@threadIdx().x;  # thread ID, dimension x
            $ty = (ParallelStencil.ParallelKernel.@blockIdx().y-1) * ParallelStencil.ParallelKernel.@blockDim().y + ParallelStencil.ParallelKernel.@threadIdx().y;  # thread ID, dimension y
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
            $tx = (ParallelStencil.ParallelKernel.@blockIdx().x-1) * ParallelStencil.ParallelKernel.@blockDim().x + ParallelStencil.ParallelKernel.@threadIdx().x;  # thread ID, dimension x
            $ty = (ParallelStencil.ParallelKernel.@blockIdx().y-1) * ParallelStencil.ParallelKernel.@blockDim().y + ParallelStencil.ParallelKernel.@threadIdx().y;  # thread ID, dimension y
            $tz = (ParallelStencil.ParallelKernel.@blockIdx().z-1) * ParallelStencil.ParallelKernel.@blockDim().z + ParallelStencil.ParallelKernel.@threadIdx().z;  # thread ID, dimension z
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
promote_ranges(ranges)                          = @ArgumentError("ranges must be a Tuple of UnitRange of size 1, 2 or 3 (obtained: $ranges; its type is: $(typeof(ranges))).")

promote_maxsize(maxsize::MAXSIZE_TYPE_1D)       = (maxsize,    1, 1)
promote_maxsize(maxsize::MAXSIZE_TYPE_1D_TUPLE) = (maxsize..., 1, 1)
promote_maxsize(maxsize::MAXSIZE_TYPE_2D)       = (maxsize..., 1)
promote_maxsize(maxsize::MAXSIZE_TYPE)          = maxsize
promote_maxsize(maxsize)                        = @ArgumentError("maxsize must be a Tuple of Integer of size 1, 2 or 3 (obtained: $maxsize; its type is: $(typeof(maxsize))).")

maxsize(t::T) where T<:Union{Tuple, NamedTuple} = maxsize(t...)
maxsize(A::T) where T<:AbstractArray            = (size(A,1),size(A,2),size(A,3))          # NOTE: using size(A,dim) three times instead of size(A) ensures to have a tuple of length 3.
maxsize(a::T) where T<:Number                   = (1, 1, 1)
maxsize(x)                                      = _maxsize(Val{isbitstype(typeof(x))})
_maxsize(::Type{Val{true}})                     = (1, 1, 1)
_maxsize(::Type{Val{false}})                    = @ArgumentError("automatic detection of ranges not possible in @parallel <kernelcall>: some kernel arguments are neither arrays nor scalars nor any other bitstypes nor (named) tuple containing any of the former. Specify ranges or nthreads and nblocks manually.")
maxsize(x, args...)                             = merge(maxsize(x), maxsize(args...))      # NOTE: maxsize is implemented as a recursive function, which results in optimal code; otherwise, the function is not performance-negligable for small problems.
merge(a::Tuple, b::Tuple)                       = max.(a,b)

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


## FUNCTIONS TO CREATE KERNEL LAUNCH AND SYNCHRONIZATION CALLS

function create_gpu_call(package::Symbol, nblocks::Union{Symbol,Expr}, nthreads::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool, stream::Union{Symbol,Expr}, shmem::Union{Symbol,Expr,Nothing}, launch::Bool)
    synccall = async ? :(begin end) : create_synccall(package, stream)
    backend_kwargs_expr = (backend_kwargs_expr...,)
    if launch
        if !isnothing(shmem)
            if     (package == PKG_CUDA)   shmem_expr = :(shmem = $shmem)
            elseif (package == PKG_AMDGPU) shmem_expr = :(shmem = $shmem)
            else                           @ModuleInternalError("unsupported GPU package (obtained: $package).")
            end
            backend_kwargs_expr = (backend_kwargs_expr..., shmem_expr) 
        end
        if     (package == PKG_CUDA)   return :( CUDA.@cuda blocks=$nblocks threads=$nthreads stream=$stream $(backend_kwargs_expr...) $kernelcall; $synccall )
        elseif (package == PKG_AMDGPU) return :( AMDGPU.@roc gridsize=$nblocks groupsize=$nthreads stream=$stream $(backend_kwargs_expr...) $kernelcall; $synccall )
        else                           @ModuleInternalError("unsupported GPU package (obtained: $package).")
        end
    else
        if     (package == PKG_CUDA)   return :( CUDA.@cuda  launch=false $(backend_kwargs_expr...) $kernelcall)  # NOTE: runtime arguments must be omitted when the kernel is not launched (backend_kwargs_expr must not contain any around time argument)
        elseif (package == PKG_AMDGPU) return :( AMDGPU.@roc launch=false $(backend_kwargs_expr...) $kernelcall)  # NOTE: ...
        else                           @ModuleInternalError("unsupported GPU package (obtained: $package).")
        end
    end
end

function create_synccall(package::Symbol, stream::Union{Symbol,Expr})
    if     (package == PKG_CUDA)   synchronize_cuda(stream)
    elseif (package == PKG_AMDGPU) synchronize_amdgpu(stream)
    else                           @ModuleInternalError("unsupported GPU package (obtained: $package).")
    end
end

function default_stream(package)
    if     (package == PKG_CUDA)    return :(CUDA.stream()) # Use the default stream of the task.
    elseif (package == PKG_AMDGPU)  return :(AMDGPU.stream()) # Use the default stream of the task.
    else                            @ModuleInternalError("unsupported GPU package (obtained: $package).")
    end
end