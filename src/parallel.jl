import .ParallelKernel: get_name, set_name, get_body, set_body!, add_return, remove_return, extract_kwargs, split_parallel_args, extract_tuple, substitute, literaltypes, push_to_signature!, add_loop, add_threadids, promote_maxsize

# NOTE: @parallel and @parallel_indices and @parallel_async do not appear in the following as they are extended and therefore re-defined here in parallel.jl
@doc replace(ParallelKernel.SYNCHRONIZE_DOC,        "@init_parallel_kernel" => "@init_parallel_stencil") macro synchronize(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.@synchronize($(args...)))); end


const PARALLEL_DOC = """
    @parallel kernel
    @parallel inbounds=... memopt=... ndims=... kernel

Declare the `kernel` parallel and containing stencil computations be performed with one of the submodules `ParallelStencil.FiniteDifferences{1D|2D|3D}` (or with a compatible custom module or set of macros).

# Optional keyword arguments
- `inbounds::Bool`: whether to apply `@inbounds` to the kernel. The default is `false` or as set with the `inbounds` keyword argument of [`@init_parallel_stencil`](@ref).
- `memopt::Bool=false`: whether to perform advanced stencil-specific on-chip memory optimisations. If `memopt=true` is set, then it must also be set in the corresponding kernel call(s).
!!! note "Advanced optional keyword arguments"
    - `ndims::Integer|Tuple`: the number of dimensions used for the stencil computations in the kernels: 1, 2 or 3 (or a tuple containing any of the previous in order to generate a method for each of the given values - this can only work correctly if the macros used *and loaded* work for any of the chosen values of `ndims`!). A default can be set with the `ndims` keyword argument of [`@init_parallel_stencil`](@ref). The keyword argument `N` becomes mandatory when `ndims` is a tuple in order to dispatch on the number of dimensions (see below).
    - `N::Integer|Tuple`: the value(s) a type parameter `N` in the kernel method signatures must take. The values are typically computed based on `ndims` (set with the corresponding keyword argument of the `@parallel` macro or `@init_parallel_stencil`), which will be substituted in the expression before evaluating it. This enables dispatching on the number of dimensions in the kernel methods (e.g., `@parallel ndims=(1,3) N=ndims function f(A::Data.Array{N}) ... end`). The keyword argument `N` is mandatory if `ndims` is a tuple and must then furthermore be a tuple of the same length as `ndims`.

See also: [`@init_parallel_stencil`](@ref)

--------------------------------------------------------------------------------
    @parallel kernelcall
    @parallel memopt=... kernelcall
    @parallel ∇=... kernelcall
    @parallel ∇=... memopt=... kernelcall

!!! note "Advanced"
        @parallel ranges kernelcall
        @parallel nblocks nthreads kernelcall
        @parallel ranges nblocks nthreads kernelcall
        @parallel (...) memopt=... configcall=... backendkwargs... kernelcall
        @parallel ∇=... ad_mode=... ad_annotations=... (...) memopt=... backendkwargs... kernelcall

Declare the `kernelcall` parallel. The kernel will automatically be called as required by the package for parallelization selected with [`@init_parallel_kernel`](@ref). Synchronizes at the end of the call (if a stream is given via keyword arguments, then it synchronizes only this stream). The keyword argument `∇` triggers a parallel call to the gradient kernel instead of the kernel itself. The automatic differentiation is performed with the package Enzyme.jl (refer to the corresponding documentation for Enzyme-specific terms used below); Enzyme needs to be imported before ParallelStencil in order to have it load the corresponding extension.

!!! note "Runtime hardware selection"
    When KernelAbstractions is initialized, this wrapper consults [`current_hardware`](@ref) to determine the runtime hardware target. The symbol defaults to `:cpu` and can be switched to select other targets via [`select_hardware`](@ref).

# Arguments
- `kernelcall`: a call to a kernel that is declared parallel.
!!! note "Advanced optional arguments"
    - `ranges::Tuple{UnitRange{},UnitRange{},UnitRange{}} | Tuple{UnitRange{},UnitRange{}} | Tuple{UnitRange{}} | UnitRange{}`: the ranges of indices in each dimension for which computations must be performed.
    - `nblocks::Tuple{Integer,Integer,Integer}`: the number of blocks to be used if the package CUDA, AMDGPU or Metal was selected with [`@init_parallel_kernel`](@ref).
    - `nthreads::Tuple{Integer,Integer,Integer}`: the number of threads to be used if the package CUDA, AMDGPU or Metal was selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
- `memopt::Bool=false`: whether the kernel to be launched was generated with `memopt=true` (meaning the keyword was set in the kernel declaration).
!!! note "Advanced"
    - `∇`: the variable(s) with respect to which the kernel is to be differentiated automatically and a duplicate for each variable to store the result in, separated by `->`, e.g., `∇=(A->Ā, B->B̄)`. Setting this keyword triggers a parallel call to the gradient kernel instead of the kernel itself. The duplicate variables are by default passed to Enzyme with the annotation `DuplicatedNoNeed`, e.g., `DuplicatedNoNeed(A, Ā)`. Use the keyword argument `ad_annotations` to modify this behavior.
    - `ad_mode=Enzyme.Reverse`: the automatic differentiation mode (see the documentation of Enzyme.jl for more information).
    - `ad_annotations=()`: Enzyme variable annotations for automatic differentiation in the format `(<keyword>=<variable(s)>, <keyword>=<variable(s)>, ...)`, where `<variable(s)>` can be a single variable or a tuple of variables (e.g., `ad_annotations=(Duplicated=B, Active=(a,b))`). Currently supported annotations are: $(keys(AD_SUPPORTED_ANNOTATIONS)).
    - `configcall=kernelcall`: a call to a kernel that is declared parallel, which is used for determining the kernel launch parameters. This keyword is useful, e.g., for generic automatic differentiation using the low-level submodule [`AD`](@ref).
    - `backendkwargs...`: keyword arguments to be passed further to CUDA.jl, AMDGPU.jl or Metal.jl (ignored for Threads and Polyester).

!!! note "Performance note"
    Kernel launch parameters are automatically defined with heuristics, where not defined with optional kernel arguments. For CUDA and AMDGPU, `nthreads` is typically set to (32,8,1) and `nblocks` accordingly to ensure that enough threads are launched.

See also: [`@init_parallel_kernel`](@ref)
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...)); end


##
const PARALLEL_INDICES_DOC = """
    @parallel_indices indices kernel
    @parallel_indices indices inbounds=... memopt=... ndims=... kernel

Declare the `kernel` parallel and generate the given parallel `indices` inside the `kernel` using the package for parallelization selected with [`@init_parallel_stencil`](@ref).

!!! note "Runtime hardware selection"
    When KernelAbstractions is initialized, this wrapper consults [`current_hardware`](@ref) to determine the runtime hardware target. The symbol defaults to `:cpu` and can be switched to select other targets via [`select_hardware`](@ref).

# Optional keyword arguments
    - `inbounds::Bool`: whether to apply `@inbounds` to the kernel. The default is `false` or as set with the `inbounds` keyword argument of [`@init_parallel_stencil`](@ref).
    - `memopt::Bool=false`: whether to perform advanced stencil-specific on-chip memory optimisations. If `memopt=true` is set, then it must also be set in the corresponding kernel call(s).
    !!! note "Advanced optional keyword arguments"
        - `ndims::Integer|Tuple`: the number of indexing dimensions desired when using splat syntax for the `indices`: 1, 2, 3 (a default `ndims` value can be set with the corresponding keyword argument of [`@init_parallel_stencil`](@ref)) or a tuple containing any of the previous in order to generate a method for each of the given `ndims` values. Concretely, the splat syntax (e.g., `@parallel_indices (I...) ndims=(2,3) ...`) generates a tuple of parallel indices (`I` in this example) where the length is given by the `ndims` value (here `2` for the first method and `3` for the second). This makes it possible to write kernels that are agnostic to the number of dimensions (writing, e.g., `A[I...]` to access elements of the array `A`). The keyword argument `N` becomes mandatory when `ndims` is a tuple in order to dispatch on the number of dimensions (see below).
        - `N::Integer|Tuple`: the value(s) a type parameter `N` in the kernel method signatures must take. The values are typically computed based on `ndims` (set with the corresponding keyword argument of the `@parallel_indices` macro or `@init_parallel_stencil`), which will be substituted in the expression before evaluating it. This enables dispatching on the number of dimensions in the kernel methods (e.g., `@parallel_indices (I...) ndims=(1,3) N=ndims function f(A::Data.Array{N}) ... end`). The keyword argument `N` is mandatory if `ndims` is a tuple and must then furthermore be a tuple of the same length as `ndims`.

See also: [`@init_parallel_stencil`](@ref)
"""
@doc PARALLEL_INDICES_DOC
macro parallel_indices(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...)); end


const PARALLEL_ASYNC_DOC = """
$(replace(ParallelKernel.PARALLEL_ASYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_ASYNC_DOC
macro parallel_async(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)              check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_amdgpu(args...)            check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_metal(args...)             check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_threads(args...)           check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_polyester(args...)         check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_POLYESTER)); end
macro parallel_indices_cuda(args...)      check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_indices_amdgpu(args...)    check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_indices_metal(args...)     check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_indices_threads(args...)   check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_indices_polyester(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_POLYESTER)); end
macro parallel_async_cuda(args...)        check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_async_amdgpu(args...)      check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_async_metal(args...)       check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_async_threads(args...)     check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_async_polyester(args...)   check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_POLYESTER)); end


## ARGUMENT CHECKS

function checkargs_parallel(args...)
    posargs, = split_args(args)
    if isempty(posargs) @ArgumentError("arguments missing.") end
    if is_kernel(args[end])  # Case: @parallel kernel
        if (length(posargs) != 1) @ArgumentError("wrong number of (positional) arguments in @parallel kernel call.") end
        kernel = args[end]
        if length(extract_kernel_args(kernel)[2]) > 0 @ArgumentError("keyword arguments are not allowed in the signature of @parallel kernels.") end
    elseif is_call(args[end])  # Case: @parallel <args...> kernelcall
        ParallelKernel.checkargs_parallel(args...)
    else
        @ArgumentError("the last argument must be a kernel definition or a kernel call (obtained: $(args[end])).")
    end
end

function checkargs_parallel_indices(args...)
    posargs, = split_args(args)
    indices = posargs[1]
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    ParallelKernel.checkargs_parallel_indices(posargs...)
end


## GATEWAY FUNCTIONS

parallel_async(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller)) = parallel(source, caller, args...; package=package, async=true)

function parallel(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller), async::Bool=false)
    if is_kernel(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
        kwargs = extract_kwargs(caller, kwargs_expr, (:ndims, :N, :inbounds, :padding, :memopt, :optvars, :loopdim, :loopsize, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module, :metadata_function), "@parallel <kernel>"; eval_args=(:ndims, :inbounds, :padding, :memopt, :loopdim, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module))
        ndims = haskey(kwargs, :ndims) ? kwargs.ndims : get_ndims(caller)
        is_parallel_kernel = true
        if typeof(ndims) <: Tuple
            expand_ndims_tuple(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
        else
            if haskey(kwargs, :N)
                substitute_N(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
            else
                numbertype = get_numbertype(caller)
                if !haskey(kwargs, :metadata_module)
                    get_name(kernelarg)
                    metadata_module, metadata_function = create_metadata_storage(source, caller, kernelarg)
                else
                    metadata_module, metadata_function = kwargs.metadata_module, kwargs.metadata_function
                end
                parallel_kernel(metadata_module, metadata_function, caller, package, ndims, numbertype, kernelarg, posargs...; kwargs)
            end
        end
    elseif is_call(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args)
        kwargs, backend_kwargs_expr = extract_kwargs(caller, kwargs_expr, (:memopt, :configcall, :∇, :ad_mode, :ad_annotations), "@parallel <kernelcall>", true; eval_args=(:memopt,))
        memopt                = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
        configcall            = haskey(kwargs, :configcall) ? kwargs.configcall : kernelarg
        configcall_kwarg_expr = :(configcall=$configcall)
        is_ad_highlevel       = haskey(kwargs, :∇)
        if !is_ad_highlevel && (haskey(kwargs, :ad_mode) || haskey(kwargs, :ad_annotations)) @IncoherentArgumentError("incoherent arguments `ad_mode`/`ad_annotations` in @parallel call: AD keywords are only valid if automatic differentiation is triggered with the keyword argument `∇`.") end
        if is_ad_highlevel
            ParallelKernel.parallel_call_ad(caller, kernelarg, backend_kwargs_expr, async, package, posargs, kwargs)
        elseif memopt
            if (length(posargs) > 1) @ArgumentError("maximum one positional argument (ranges) is allowed in a @parallel memopt=true call.") end
            parallel_call_memopt(caller, posargs..., kernelarg, backend_kwargs_expr, async; kwargs...)
        else
            ParallelKernel.parallel(caller, posargs..., backend_kwargs_expr..., configcall_kwarg_expr, kernelarg; package=package, async=async)
        end
    end
end


function parallel_indices(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller))
    is_parallel_kernel = false
    numbertype = get_numbertype(caller)
    posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
    kwargs = extract_kwargs(caller, kwargs_expr, (:ndims, :N, :inbounds, :padding, :memopt, :optvars, :loopdim, :loopsize, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module, :metadata_function), "@parallel_indices"; eval_args=(:ndims, :inbounds, :padding, :memopt, :loopdim, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module))
    indices_expr = posargs[1]
    ndims = haskey(kwargs, :ndims) ? kwargs.ndims : get_ndims(caller)
    if typeof(ndims) <: Tuple
        expand_ndims_tuple(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
    else
        if haskey(kwargs, :N)
            substitute_N(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
        elseif is_splatarg(indices_expr)
            parallel_indices_splatarg(caller, package, ndims, kwargs_expr, posargs..., kernelarg; kwargs)
        else
            if !haskey(kwargs, :metadata_module)
                get_name(kernelarg)
                metadata_module, metadata_function = create_metadata_storage(source, caller, kernelarg)
            else
                metadata_module, metadata_function = kwargs.metadata_module, kwargs.metadata_function
            end
            inbounds = haskey(kwargs, :inbounds) ? kwargs.inbounds : get_inbounds(caller)
            padding  = haskey(kwargs, :padding)  ? kwargs.padding  : get_padding(caller)
            memopt   = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
            if memopt
                quote
                    $(parallel_indices_memopt(metadata_module, metadata_function, is_parallel_kernel, caller, package, posargs..., kernelarg; kwargs...))  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from call: package, numbertype, )
                    $metadata_function
                end
            else
                kwargs_expr = (:(inbounds=$inbounds), :(padding=$padding))
                ParallelKernel.parallel_indices(caller, posargs..., kwargs_expr..., kernelarg; package=package)
            end
        end
    end
end


## @PARALLEL KERNEL FUNCTIONS

function expand_ndims_tuple(caller::Module, ndims::Tuple, is_parallel_kernel::Bool, kernel::Expr, kwargs::NamedTuple, posargs...)
    macroname = (is_parallel_kernel) ? "@parallel" : "@parallel_indices"
    if !(typeof(ndims) <: NTuple{N,<:Integer} where N) @KeywordArgumentError("$macroname: keyword argument 'ndims' must be an integer or a tuple of integers (obtained: $ndims).") end
    if !haskey(kwargs, :N) @KeywordArgumentError("$macroname: keyword argument 'N' is mandatory when 'ndims' is a tuple ('N' must also be present as type parameter in the function signature enabling to dispatch on). ") end
    N = eval_arg(caller, substitute(kwargs.N, :ndims, ndims))
    if !(typeof(N) <: NTuple{length(ndims),<:Integer}) @KeywordArgumentError("$macroname: keyword argument 'N' must be a tuple of integers of the same length as 'ndims' when 'ndims' is a tuple (obtained: N=$N, ndims=$ndims).") end
    kwargs_expr = (:($key=$(getproperty(kwargs, key))) for key in keys(kwargs) if key ∉ (:ndims, :N))
    if (is_parallel_kernel) ndims_methods_expr = (:(@parallel         $(posargs...) ndims=$i N=$n $(kwargs_expr...) $kernel) for (i,n) in zip(ndims,N))
    else                    ndims_methods_expr = (:(@parallel_indices $(posargs...) ndims=$i N=$n $(kwargs_expr...) $kernel) for (i,n) in zip(ndims,N))
    end
    return quote $(ndims_methods_expr...) end
end

function substitute_N(caller::Module, ndims::Integer, is_parallel_kernel::Bool, kernel::Expr, kwargs::NamedTuple, posargs...)
    macroname = (is_parallel_kernel) ? "@parallel" : "@parallel_indices"
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("$macroname: keyword argument 'ndims' is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    if !haskey(kwargs, :N) @ModuleInternalError("$macroname: substitute_N: function should never be called if keyword argument 'N' is not present.") end
    N = eval_arg(caller, substitute(kwargs.N, :ndims, ndims))
    if !(typeof(N) <: Integer) @KeywordArgumentError("$macroname: keyword argument 'N' must be an integer (or a tuple if 'ndims' is a tuple; obtained: $N).") end
    kwargs_expr = (:($key=$(getproperty(kwargs, key))) for key in keys(kwargs) if key != :N)
    if inexpr_walk(splitdef(kernel)[:whereparams], :N) @IncoherentArgumentError("$macroname: 'N' must not appear in the where clause of the kernel signature when the keyword argument 'N' is used.") end
    kernel = substitute_in_kernel(kernel, :N, N; signature_only=true, typeparams_only=true)
    if (is_parallel_kernel) return :(@parallel         $(posargs...) $(kwargs_expr...) $kernel)
    else                    return :(@parallel_indices $(posargs...) $(kwargs_expr...) $kernel)
    end
end

function parallel_indices_splatarg(caller::Module, package::Symbol, ndims::Integer, kwargs_expr, alias_indices::Expr, kernel::Expr; kwargs::NamedTuple)
    if !@capture(alias_indices, (I_...)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("@parallel_indices: keyword argument 'ndims' is required for the syntax `@parallel_indices I...` and is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    indices = get_indices_expr(ndims).args
    indices_expr = Expr(:tuple, indices...)
    kernel = macroexpand(caller, kernel)
    kernel = substitute(kernel, I, indices_expr)
    return :(@parallel_indices $indices_expr $(kwargs_expr...) $kernel)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from signature: package::Symbol, numbertype::DataType, )
end

function parallel_indices_memopt(metadata_module::Module, metadata_function::Expr, is_parallel_kernel::Bool, caller::Module, package::Symbol, indices::Union{Symbol,Expr}, kernel::Expr; ndims::Integer=get_ndims(caller), inbounds::Bool=get_inbounds(caller), padding::Bool=get_padding(caller), memopt::Bool=get_memopt(caller), optvars::Union{Expr,Symbol}=Symbol(""), loopdim::Integer=determine_loopdim(indices), loopsize::Integer=compute_loopsize(package), optranges::Union{Nothing, NamedTuple{t, <:NTuple{N,NTuple{3,UnitRange}} where N} where t}=nothing, useshmemhalos::Union{Nothing, NamedTuple{t, <:NTuple{N,Bool} where N} where t}=nothing, optimize_halo_read::Bool=true)
    if (!memopt) @ModuleInternalError("parallel_indices_memopt: called with `memopt=false` which should never happen.") end
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    if (!isa(optvars,Symbol) && !isa(optvars.head,Symbol)) @KeywordArgumentError("@parallel_indices: keyword argument 'optvars' must be a tuple of optvars or a single optvar (e.g. (A, B, C) or A ).") end
    body = get_body(kernel)
    body = remove_return(body)
    body = add_memopt(metadata_module, is_parallel_kernel, caller, package, body, indices, optvars, loopdim, loopsize, optranges, useshmemhalos, optimize_halo_read)
    body = add_return(body, package)
    set_body!(kernel, body)
    indices = extract_tuple(indices)
    return :(@parallel_indices $(Expr(:tuple, indices[1:end-1]...)) ndims=$ndims inbounds=$inbounds padding=$padding memopt=false metadata_module=$metadata_module metadata_function=$metadata_function $kernel)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from signature: package::Symbol, numbertype::DataType, )
end

function parallel_kernel(metadata_module::Module, metadata_function::Expr, caller::Module, package::Symbol, ndims::Integer, numbertype::DataType, kernel::Expr; kwargs::NamedTuple)
    is_parallel_kernel = true
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("@parallel: keyword argument 'ndims' is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    inbounds = haskey(kwargs, :inbounds) ? kwargs.inbounds : get_inbounds(caller)
    padding  = haskey(kwargs, :padding)  ? kwargs.padding  : get_padding(caller)
    memopt = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
    indices = get_indices_expr(ndims).args
    indices_dir = get_indices_dir_expr(ndims).args
    body = get_body(kernel)
    body = remove_return(body)
    validate_body(body)
    kernelargs = splitarg.(extract_kernel_args(kernel)[1])
    argvars = (arg[1] for arg in kernelargs)
    check_mask_macro(caller)
    onthefly_vars, onthefly_exprs, write_vars, body = extract_onthefly_arrays!(body, argvars)
    has_onthefly = !isempty(onthefly_vars)
    body = apply_masks(body, indices)
    body = macroexpand(caller, body)
    body = handle_padding(caller, body, padding, indices; handle_view_accesses=false, delay_dir_handling=has_onthefly && padding) # NOTE: delay_dir_handling is mandatory in case of on-the-fly with padding, because the macros (missing dir_handling) created will only be available in the next world age.
    if has_onthefly
        onthefly_syms  = gensym_world.(onthefly_vars, (@__MODULE__,))
        onthefly_exprs = macroexpand.((caller,), onthefly_exprs)
        onthefly_exprs = handle_padding.((caller,), onthefly_exprs, (padding,), (indices,); handle_view_accesses=false, dir_handling=!padding) # NOTE: dir_handling is done after macro expansion with the delayed handling.
        onthefly_exprs = insert_onthefly!.(onthefly_exprs, (onthefly_vars,), (onthefly_syms,), (indices,), (indices_dir,))
        onthefly_exprs = handle_padding.((caller,), onthefly_exprs, (padding,), (indices,); handle_indexing=false)
        body           = insert_onthefly!(body, onthefly_vars, onthefly_syms, indices, indices_dir)
        create_onthefly_macro.((caller,), onthefly_syms, onthefly_exprs, onthefly_vars, (indices,), (indices_dir,))
    end
    body = handle_padding(caller, body, padding, indices; handle_indexing=false)
    if isgpu(package) kernel = insert_device_types(caller, kernel) end
    if !memopt
        kernel = adjust_signatures(kernel, package)
        body   = handle_inverses(body)
        body   = handle_indices_and_literals(body, indices, package, numbertype)
        if (inbounds) body = add_inbounds(body) end
    end
    body = add_return(body, package)
    set_body!(kernel, body)
    if memopt
        expanded_kernel = macroexpand(caller, kernel)
        quote
            $(parallel_indices_memopt(metadata_module, metadata_function, is_parallel_kernel, caller, package, get_indices_expr(ndims), expanded_kernel; kwargs...)) #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from call: package, numbertype, )
            $metadata_function
        end
    else
        if package == PKG_KERNELABSTRACTIONS
            kernel = :(KernelAbstractions.@kernel $kernel)
        end
        return kernel # TODO: later could be here called parallel_indices instead of adding the threadids etc above.
    end
end


## @PARALLEL CALL FUNCTIONS

function parallel_call_memopt(caller::Module, ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    if haskey(backend_kwargs_expr, :shmem) @KeywordArgumentError("@parallel <kernelcall>: keyword `shmem` is not allowed when memopt=true is set.") end
    package             = get_package(caller)
    nthreads_x_max      = ParallelKernel.determine_nthreads_x_max(package)
    nthreads_max_memopt = determine_nthreads_max_memopt(package)
    configcall_kwarg_expr = :(configcall=$configcall)
    metadata_call   = create_metadata_call(configcall)
    metadata_module = metadata_call
    stencilranges   = :($(metadata_module).stencilranges)
    use_shmemhalos  = :($(metadata_module).use_shmemhalos)
    optvars         = :($(metadata_module).optvars)
    loopdim          = :($(metadata_module).loopdim)
    loopsize        = :($(metadata_module).loopsize)
    loopsizes       = :(($loopdim==3) ? (1, 1, $loopsize) : ($loopdim==2) ? (1, $loopsize, 1) : ($loopsize, 1, 1))
    maxsize         = :(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges($ranges)), $loopsizes))
    nthreads        = :( ParallelStencil.compute_nthreads_memopt($nthreads_x_max, $nthreads_max_memopt, $maxsize, $loopdim, $stencilranges) )
    nblocks         = :( ParallelStencil.ParallelKernel.compute_nblocks($maxsize, $nthreads) )
    numbertype      = get_numbertype(caller) # not :(eltype($(optvars)[1])) # TODO: see how to obtain number type properly for each array: the type of the call call arguments corresponding to the optimization variables should be checked
    dim1 = :(($loopdim==3) ? 1 : ($loopdim==2) ? 1 : 2) # TODO: to be determined if that is what is desired for loopdim 1 and 2.
    dim2 = :(($loopdim==3) ? 2 : ($loopdim==2) ? 3 : 3) # TODO: to be determined if that is what is desired for loopdim 1 and 2.
    A = gensym("A")
    shmem = :(sum(($nthreads[$dim1]+$use_shmemhalos[$A]*(length($(stencilranges)[$A][$dim1])-1))*($nthreads[$dim2]+$use_shmemhalos[$A]*(length($(stencilranges)[$A][$dim2])-1))*sizeof($numbertype) for $A in $optvars))
    if (async) return :(@parallel_async memopt=false $configcall_kwarg_expr $ranges $nblocks $nthreads shmem=$shmem $(backend_kwargs_expr...) $kernelcall)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs
    else       return :(@parallel       memopt=false $configcall_kwarg_expr $ranges $nblocks $nthreads shmem=$shmem $(backend_kwargs_expr...) $kernelcall)  #TODO: ...
    end
end

function parallel_call_memopt(caller::Module, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    package             = get_package(caller)
    nthreads_x_max      = ParallelKernel.determine_nthreads_x_max(package)
    nthreads_max_memopt = determine_nthreads_max_memopt(package)
    metadata_call       = create_metadata_call(configcall)
    metadata_module     = metadata_call
    loopdim             = :($(metadata_module).loopdim)
    is_parallel_kernel  = :($(metadata_module).is_parallel_kernel)
    ranges              = :( ($is_parallel_kernel) ? ParallelStencil.get_ranges_memopt($nthreads_x_max, $nthreads_max_memopt, $loopdim, $(configcall.args[2:end]...)) : ParallelStencil.ParallelKernel.get_ranges($(configcall.args[2:end]...)))
    parallel_call_memopt(caller, ranges, kernelcall, backend_kwargs_expr, async; memopt=memopt, configcall=configcall)
end


## FUNCTIONS FOR APPLYING OPTIMISATIONS

function add_memopt(metadata_module::Module, is_parallel_kernel::Bool, caller::Module, package::Symbol, body::Expr, indices::Union{Symbol,Expr}, optvars::Union{Expr,Symbol}, loopdim::Integer, loopsize::Integer, optranges::Union{Nothing, NamedTuple{t, <:NTuple{N,NTuple{3,UnitRange}} where N} where t}, useshmemhalos::Union{Nothing, NamedTuple{t, <:NTuple{N,Bool} where N} where t}, optimize_halo_read::Bool)
    memopt(metadata_module, is_parallel_kernel, caller, indices, optvars, loopdim, loopsize, optranges, useshmemhalos, optimize_halo_read, body; package=package)
end


## FUNCTIONS TO DETERMINE OPTIMIZATION PARAMETERS

determine_nthreads_max_memopt(package::Symbol)  = (package == PKG_AMDGPU) ? NTHREADS_MAX_MEMOPT_AMDGPU : ((package == PKG_CUDA) ? NTHREADS_MAX_MEMOPT_CUDA : NTHREADS_MAX_MEMOPT_METAL)
determine_loopdim(indices::Union{Symbol,Expr}) = isa(indices,Expr) && (length(indices.args)==3) ? 3 : LOOPDIM_NONE # TODO: currently only loopdim=3 is supported.

function compute_loopsize(package::Symbol)
    compute_capability = get_compute_capability(package)
    if compute_capability == v"∞" # if not set (could also be not CUDA), choose a value that should work well for all architectures, favouring newer ones.
        return 32
    elseif compute_capability < v"8"
        return 16
    elseif compute_capability < v"9"
        return 32
    else
        return 64
    end
end


## FUNCTIONS TO COMPUTE NTHREADS, NBLOCKS

function compute_nthreads_memopt(nthreads_x_max, nthreads_max_memopt, maxsize, loopdim, stencilranges) # This is a heuristic, which results typcially in (32,4,1) threads for a 3-D case.
    maxsize = promote_maxsize(maxsize)
    nthreads = ParallelKernel.compute_nthreads(maxsize; nthreads_x_max=nthreads_x_max, nthreads_max=nthreads_max_memopt, flatdim=loopdim)
    for stencilranges_A in values(stencilranges)
        haloextensions = ((length(stencilranges_A[1])-1)*(loopdim!=1), (length(stencilranges_A[2])-1)*(loopdim!=2), (length(stencilranges_A[3])-1)*(loopdim!=3))
        if (2*prod(nthreads) < prod(nthreads .+ haloextensions)) @ArgumentError("@parallel <kernelcall>: the automatic determination of nthreads is not possible for this case. Please specify `nthreads` and `nblocks`.")  end # NOTE: this is a simple heuristic to compute compare the number of threads to the total number of cells including halo.
    end
    # TODO: check if this can simply be removed or even the kernel something need to be adapted:
    if any(maxsize .% nthreads .!= 0) @ArgumentError("@parallel <kernelcall>: memopt optimization not possible for the given maximum array size in the kernel arguments (the maximum array size must be dividable without rest by the number of threads per block)") end # NOTE: this is a requirement for the reading into shared memory because the re-indexing requires that no thread aborts the kernel early. A way around it, is to specify the range such that the condition verified here is true (meaning the range length must be dividable by the number of threads without rest). This can be done automatically for @parallel kernels, because there the array bounds are always verified, but it must be done explicitly by the user for @parallel_indices kernels.
    return nthreads
end

function get_ranges_memopt(nthreads_x_max, nthreads_max_memopt, loopdim, args...)
    ranges   = ParallelKernel.get_ranges(args...)
    maxsize  = length.(ranges)
    nthreads = ParallelKernel.compute_nthreads(maxsize; nthreads_x_max=nthreads_x_max, nthreads_max=nthreads_max_memopt, flatdim=loopdim)
    # TODO: the following code reduces performance from ~482 GB/s to ~478 GB/s
    rests    = maxsize .% nthreads
    ranges_adjustment = ( (rests[1] != 0) ? (nthreads[1] - rests[1]) : 0,
                          (rests[2] != 0) ? (nthreads[2] - rests[2]) : 0,
                          (rests[3] != 0) ? (nthreads[3] - rests[3]) : 0 )
    ranges = ParallelKernel.compute_ranges(maxsize .+ ranges_adjustment) # NOTE: this makes memopt possible also if the maximum array size is not dividable without rest by the number of threads; however, it requires that all array accesses in the kernel are bounds checked. For parallel indices kernels the user has to guarantee that himself.
    return ranges
end


## FUNCTIONS TO DEAL WITH MASKS (@WITHIN) AND INDICES

is_splatarg(x) = isa(x,Expr) && (x.head == :...)

function check_mask_macro(caller::Module)
    if !isdefined(caller, Symbol("@within")) @MethodPluginError("the macro @within is not defined in the caller. You need to load one of the submodules ParallelStencil.FiniteDifferences{1|2|3}D (or a compatible custom module or set of macros).") end
    methods_str = string(methods(getfield(caller, Symbol("@within"))))
    if !occursin(r"(var\"@within\"|@within)\(__source__::LineNumberNode, __module__::Module, .*::String, .*\)", methods_str) @MethodPluginError("the signature of the macro @within is not compatible with ParallelStencil (detected signature: \"$methods_str\"). The signature must correspond to the description in ParallelStencil.WITHIN_DOC. See in ParallelStencil.FiniteDifferences{1|2|3}D for examples.") end
end

function apply_masks(expr::Expr, indices::Array{Any}; do_shortif=false)
    args = expr.args
    for i=1:length(args)
        if typeof(args[i]) == Expr
            e = args[i]
            if e.head == :(=) && typeof(e.args[1]) == Expr && e.args[1].head == :macrocall
                lefthand_macro = e.args[1].args[1]
                lefthand_var   = e.args[1].args[3]
                macroname = string(lefthand_macro)
                if do_shortif
                    args[i] = quote
                        $(e.args[1]) = (@within($macroname, $lefthand_var)) ? $(e.args[2]) : $(e.args[1]) #TODO: as else-variable (currently e.args[1], e.g. T2), the right value from the righthand side should be taken (e.g. T) for best perf. This requires though user indication of the kind T2==T as kwarg... Also, these shortifs can only be used if the within conditions are centered (1 < ... < size(A,1)-1) instead of as present ((0 < ... < size(A,1)-2)).
                    end
                else
                    args[i] = quote
                                if (@within($macroname, $lefthand_var))
                                    $e
                                end
                            end
                end
            else
                args[i] = apply_masks(e, indices; do_shortif=do_shortif)
            end
        end
    end
    return expr
end

function get_indices_expr(ndims::Integer)
    if ndims == 1
        return :($(INDICES[1]),)
    elseif ndims == 2
        return :($(INDICES[1]), $(INDICES[2]))
    elseif ndims == 3
        return :($(INDICES[1]), $(INDICES[2]), $(INDICES[3]))
    else
        @ModuleInternalError("argument 'ndims' must be 1, 2 or 3.")
    end
end

function get_indices_dir_expr(ndims::Integer)
    if ndims == 1
        return :($(INDICES_DIR[1]),)
    elseif ndims == 2
        return :($(INDICES_DIR[1]), $(INDICES_DIR[2]))
    elseif ndims == 3
        return :($(INDICES_DIR[1]), $(INDICES_DIR[2]), $(INDICES_DIR[3]))
    else
        @ModuleInternalError("argument 'ndims' must be 1, 2 or 3.")
    end
end


## FUNCTIONS TO CREATE METADATA STORAGE

function create_metadata_storage(source::LineNumberNode, caller::Module, kernel::Expr)
    kernelid = get_kernelid(get_name(kernel), source.file, source.line)
    create_module(caller, MOD_METADATA_PS)
    topmodule = @eval(caller, $MOD_METADATA_PS)
    create_module(topmodule, kernelid)
    metadata_module = @eval(topmodule, $kernelid)
    metadata_function = create_metadata_function(kernel, metadata_module)
    return metadata_module, metadata_function
end

function create_module(hostmodule::Module, modulename::Symbol; do_baremodule=true)
    if !isdefined(hostmodule, modulename)
        moduleexpr = (do_baremodule) ? :(baremodule $modulename end) : :(module $modulename end)
        @eval(hostmodule, $moduleexpr)
    end
end

function create_metadata_function(kernel::Expr, metadata_module::Module) # NOTE: unlike the creation of the module above, the creation of the matter data function has to happen every time: if we redefine the same function we to have to redefine the meta data...
    metadata_function = deepcopy(kernel)
    kernelname = get_name(kernel)
    functionname = get_meta_function(kernelname)
    metadata_function = set_name(metadata_function, functionname)
    set_body!(metadata_function, :(return $metadata_module))
    return metadata_function
end

function create_metadata_call(configcall::Expr)
    metadata_call = deepcopy(configcall)
    kernelname = metadata_call.args[1]
    metadata_call.args[1] = get_meta_function(kernelname)
    return metadata_call
end

get_kernelid(kernelname, file, line) = Symbol("$(kernelname)_$(file)_$(line)")
get_meta_function(kernelname)        = Symbol("$(META_FUNCTION_PREFIX)$(GENSYM_SEPARATOR)$(kernelname)")


## FUNCTIONS TO DEAL WITH ON-THE-FLY ASSIGNMENTS

function extract_onthefly_arrays!(body, argvars)
    onthefly_vars  = ()
    onthefly_exprs = ()
    write_vars     = ()
    statements     = get_statements(body)
    for statement in statements
        if is_array_assignment(statement)
            if !@capture(statement, @m_(A_) = assign_expr_) @ArgumentError(ERRMSG_KERNEL_UNSUPPORTED) end
            if any(inexpr_walk.((A,), argvars))
                write_vars = (write_vars..., A)
            end
        end
    end
    for statement in statements
        if is_array_assignment(statement)
            if !@capture(statement, @m_(A_) = assign_expr_) @ArgumentError(ERRMSG_KERNEL_UNSUPPORTED) end
            if !any(inexpr_walk.((A,), argvars))
                if (m != Symbol("@all"))         @ArgumentError("unsupported kernel statements in @parallel kernel definition: partial assignments are not possible for arrays that are not stored in global memory (arrays that are not among the arguments of the kernel); use '@all' instead.") end
                if (inexpr_walk(assign_expr, A)) @ArgumentError("unsupported kernel statements in @parallel kernel definition: auto-dependency is not possible for arrays that are not stored in global memory (arrays that are not among the arguments of the kernel).") end
                if any(inexpr_walk.((assign_expr,), write_vars)) # NOTE: in this case here could later be allocated a local array instead
                    @ArgumentError("unsupported kernel statements in @parallel kernel definition: the assignment of $A should be done on the fly as it is not among the arguments of the kernel; however, this is not possible because it depends on at least one variable that is not read-only within the scope of the kernel (any of: $write_vars).")
                else
                    onthefly_vars  = (onthefly_vars..., A)
                    onthefly_exprs = (onthefly_exprs..., assign_expr)
                    body           = substitute(body, statement, NOEXPR)
                end
            end
        end
    end
    return onthefly_vars, onthefly_exprs, write_vars, body
end

function insert_onthefly!(expr, onthefly_vars, onthefly_syms, indices::Array, indices_dir::Array)
    indices = (indices...,)
    indices_dir = (indices_dir...,)
    for (A, m) in zip(onthefly_vars, onthefly_syms)
        expr = substitute(expr, A, m, indices, indices_dir)
    end
    return expr
end

function determine_local_index_dir(local_index, dim)
    id_l = local_index
    id_l = increment_arg(id_l, INDICES_DIR_FUNCTIONS_SYMS[dim])
    id_l = substitute(id_l, INDICES_DIR[dim], :($(INDICES_DIR_FUNCTIONS_SYMS[dim])(2)))
    id_l = substitute(id_l, INDICES[dim], INDICES_DIR[dim])
    return id_l
end

function create_onthefly_macro(caller, m, expr, var, indices, indices_dir)
    ndims                 = length(indices)
    ix, iy, iz            = gensym_world.(("ix","iy","iz"), (@__MODULE__,))
    ixd, iyd, izd         = gensym_world.(("ixd","iyd","izd"), (@__MODULE__,))
    local_indices         = (ndims==3) ? (ix, iy, iz) : (ndims==2) ? (ix, iy) : (ix,)
    local_indices_dir     = (ndims==3) ? (ixd, iyd, izd) : (ndims==2) ? (ixd, iyd) : (ixd,)
    for (index, local_index) in zip(indices, local_indices)
        expr = substitute(expr, index, Expr(:$, local_index))
    end
    for (index, local_index) in zip(indices_dir, local_indices_dir)
        expr = substitute(expr, index, Expr(:$, local_index))
    end
    local_assign = quote
        $((:($(local_indices_dir[i]) = ParallelStencil.determine_local_index_dir($(local_indices[i]), $i)) for i=1:ndims)...)
    end
    expr_quoted = :($(Expr(:quote, expr)))
    m_function = :($m($(local_indices...)) = ($local_assign; $expr_quoted))
    m_macro = :(macro $m(args...) if (length(args)!=$ndims) ParallelStencil.@ArgumentError("unsupported kernel statements in @parallel kernel definition: wrong number of indices in $var (expected $ndims indices).") end; esc($m(args...)) end)
    @eval(caller, $m_function)
    @eval(caller, $m_macro)
    return
end
