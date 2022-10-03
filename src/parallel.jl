import .ParallelKernel: get_body, set_body!, add_return, remove_return, extract_kwargs, split_parallel_args, extract_tuple, substitute, literaltypes, push_to_signature!, add_loop, add_threadids

const PARALLEL_DOC = """
    @parallel kernel

Declare the `kernel` parallel and containing stencil computations be performed with one of the submodules `ParallelStencil.FiniteDifferences{1D|2D|3D}` (or with a compatible custom module or set of macros).

See also: [`@init_parallel_stencil`](@ref)

--------------------------------------------------------------------------------
$(replace(ParallelKernel.PARALLEL_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...)); end


const PARALLEL_INDICES_DOC = """
$(replace(ParallelKernel.PARALLEL_INDICES_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_INDICES_DOC
macro parallel_indices(args...) check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...)); end


const PARALLEL_ASYNC_DOC = """
$(replace(ParallelKernel.PARALLEL_ASYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_ASYNC_DOC
macro parallel_async(args...) check_initialized(); checkargs_parallel(args...); esc(parallel_async(__module__, args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)    check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_CUDA)); end
macro parallel_threads(args...) check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_THREADS)); end
macro parallel_indices_cuda(args...)    check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...; package=PKG_CUDA)); end
macro parallel_indices_threads(args...) check_initialized(); checkargs_parallel_indices(args...); esc(parallel_indices(__module__, args...; package=PKG_THREADS)); end
macro parallel_async_cuda(args...)      check_initialized(); checkargs_parallel(args...); esc(parallel_async(__module__, args...; package=PKG_CUDA)); end
macro parallel_async_threads(args...)   check_initialized(); checkargs_parallel(args...); esc(parallel_async(__module__, args...; package=PKG_THREADS)); end


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
    ParallelKernel.checkargs_parallel_indices(posargs...)
end


## GATEWAY FUNCTIONS

parallel_async(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package()) = parallel(caller, args...; package=package, async=true)

function parallel(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(), async::Bool=false)
    if is_kernel(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
        kwargs     = extract_kwargs(caller, kwargs_expr, (:loopopt, :optvars, :optdim, :loopsize, :halosize), "@parallel <kernel>"; eval_args=(:loopopt, :optdim, :halosize))
        numbertype = get_numbertype()
        ndims      = get_ndims()
        parallel_kernel(caller, package, numbertype, ndims, kernelarg, posargs...; kwargs)
    elseif is_call(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args)
        kwargs, backend_kwargs_expr = extract_kwargs(caller, kwargs_expr, (:loopopt, :optvars, :optdim, :loopsize, :halosize), "@parallel <kernelcall>", true; eval_args=(:loopopt, :optdim, :halosize))
        if haskey(kwargs, :loopopt) && kwargs.loopopt
            parallel_call_loopopt(posargs..., kernelarg, backend_kwargs_expr, async; kwargs...)
        else
            ParallelKernel.parallel(args...; package=package)
        end
    end
end


function parallel_indices(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package())
    numbertype = get_numbertype()
    posargs, kwargs_expr = split_args(args)
    kwargs = extract_kwargs(caller, kwargs_expr, (:loopopt, :optvars, :optdim, :loopsize, :halosize), "@parallel_indices"; eval_args=(:loopopt, :optdim, :halosize))
    if haskey(kwargs, :loopopt) && kwargs.loopopt
        parallel_indices_loopopt(package, numbertype, posargs...; kwargs...)
    else
        ParallelKernel.parallel_indices(posargs...; package=package)
    end
end


## @PARALLEL KERNEL FUNCTIONS

function parallel_indices_loopopt(package::Symbol, numbertype::DataType, indices::Union{Symbol,Expr}, kernel::Expr; loopopt::Bool=false, optvars::Union{Expr,Symbol}=Symbol(""), optdim::Integer=determine_optdim(), loopsize::Union{Expr,Symbol,Integer}=compute_loopsize(), halosize::Union{Integer,NTuple{N,Integer} where N}=compute_halosize(), indices_shift::Tuple{<:Integer,<:Integer,<:Integer}=(0,0,0))
    if (!loopopt) @ModuleInternalError("parallel_indices_loopopt: called with `loopopt=false` which should never happen.") end
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices or a single index (e.g. (ix, iy, iz) or (ix, iy) or ix ).") end
    indices = extract_tuple(indices)
    body = get_body(kernel)
    body = remove_return(body)
    body = add_loopopt(body, indices, optvars, optdim, loopsize, halosize, indices_shift)
    body = add_return(body)
    set_body!(kernel, body)
    return :(@parallel_indices $(Expr(:tuple, indices[1:end-1]...)) $kernel)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs
end

function parallel_kernel(caller::Module, package::Symbol, numbertype::DataType, ndims::Integer, kernel::Expr; kwargs::NamedTuple)
    loopopt = haskey(kwargs, :loopopt) && kwargs.loopopt
    indices = get_indices_expr(ndims).args
    body = get_body(kernel)
    body = remove_return(body)
    validate_body(body)
    if (package == PKG_CUDA)
        kernel = substitute(kernel, :(Data.Array),      :(Data.DeviceArray))
        kernel = substitute(kernel, :(Data.Cell),       :(Data.DeviceCell))
        kernel = substitute(kernel, :(Data.CellArray),  :(Data.DeviceCellArray))
        kernel = substitute(kernel, :(Data.TArray),     :(Data.DeviceTArray))
        kernel = substitute(kernel, :(Data.TCell),      :(Data.DeviceTCell))
        kernel = substitute(kernel, :(Data.TCellArray), :(Data.DeviceTCellArray))
    end
    if !loopopt
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
    end
    check_mask_macro(caller)
    body = apply_masks(body, indices)
    body = add_return(body)
    set_body!(kernel, body)
    if loopopt 
        expanded_kernel = macroexpand(caller, kernel)
        parallel_indices_loopopt(package, numbertype, get_indices_expr(ndims), expanded_kernel; indices_shift=(1,1,1), kwargs...)
    else
        return kernel # TODO: later could be here called parallel_indices instead of adding the threadids etc above.
    end
end


## @PARALLEL CALL FUNCTIONS

function parallel_call_loopopt(ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; loopopt::Bool=false, optvars::Union{Expr,Symbol}=Symbol(""), optdim::Integer=determine_optdim(), loopsize::Union{Expr,Symbol,Integer}=compute_loopsize(), halosize::Union{Integer,NTuple{N,Integer} where N}=compute_halosize())
    if (!loopopt) @ModuleInternalError("parallel_call_loopopt: called with `loopopt=false` which should never happen.") end
    if isa(optvars, Expr) @ArgumentError("parallel_call_loopopt: at present, one and only one variable is supported in argument `optvars`.") end
    if haskey(backend_kwargs_expr, :shmem) @KeywordArgumentError("@parallel <kernelcall>: keyword `shmem` is not allowed when loopopt=true is set.") end
    if     (optdim==3) loopsizes = :(1, 1, $loopsize)
    elseif (optdim==2) loopsizes = :(1, $loopsize, 1)
    elseif (optdim==1) loopsizes = :($loopsize, 1, 1)
    end
    maxsize   = :(cld.(length.(ParallelStencil.ParallelKernel.promote_ranges($ranges)), $loopsizes))
    nthreads  = :( ParallelStencil.compute_nthreads_loopopt($maxsize, $optdim, $halosize) )
    nblocks   = :( ParallelStencil.ParallelKernel.compute_nblocks($maxsize, $nthreads) )
    numbertype = (optvars==Symbol("")) ? get_numbertype() : :(eltype(optvars))
    if     (optdim==3) shmem = :(($nthreads[1]+2*$halosize[1])*($nthreads[2]+2*$halosize[2])*sizeof($numbertype))
    elseif (optdim==2) shmem = :(($nthreads[1]+2*$halosize[1])*($nthreads[3]+2*$halosize[3])*sizeof($numbertype)) # TODO: to be determined if that is what is desired.
    elseif (optdim==1) shmem = :(($nthreads[2]+2*$halosize[2])*($nthreads[3]+2*$halosize[3])*sizeof($numbertype)) # TODO: to be determined if that is what is desired.
    end
    if (async) return :(@parallel_async $ranges $nblocks $nthreads shmem=$shmem $(backend_kwargs_expr...) $kernelcall)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs
    else       return :(@parallel       $ranges $nblocks $nthreads shmem=$shmem $(backend_kwargs_expr...) $kernelcall)  #TODO: ...
    end
end

function parallel_call_loopopt(kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; loopopt::Bool=false, optvars::Union{Expr,Symbol}=Symbol(""), optdim::Integer=determine_optdim(), loopsize::Union{Expr,Symbol,Integer}=compute_loopsize(), halosize::Union{Integer,NTuple{N,Integer} where N}=compute_halosize())
    if (!loopopt) @ModuleInternalError("parallel_call_loopopt: called with `loopopt=false` which should never happen.") end
    ranges = :( ParallelStencil.ParallelKernel.get_ranges($(kernelcall.args[2:end]...)) )
    parallel_call_loopopt(ranges, kernelcall, backend_kwargs_expr, async; loopopt=loopopt, optvars=optvars, optdim=optdim, loopsize=loopsize, halosize=halosize)
end


## FUNCTIONS FOR APPLYING OPTIMISATIONS

function add_loopopt(body::Expr, indices::Array{<:Union{Expr,Symbol}}, optvars::Union{Expr,Symbol}, optdim::Integer, loopsize::Union{Expr,Symbol,Integer}, halosize::Union{Integer,NTuple{N,Integer} where N}, indices_shift::Tuple{<:Integer,<:Integer,<:Integer})
    if (optvars == Symbol("")) @KeywordArgumentError("@parallel <kernel>: keyword argument `optvars` is mandatory when `loopopt=true` is set.") end
    if isa(optvars, Expr) @ArgumentError("loopopt: at present, only one variable is supported in argument `optvars`.") end
    quote
        ParallelStencil.@loopopt $(Expr(:tuple, indices...)) $optvars $optdim $loopsize $(Expr(:tuple, halosize...)) $indices_shift begin
            $body
        end
    end
end


## FUNCTIONS TO DETERMINE OPTIMIZATION PARAMETERS

compute_halosize() = return (1,1,0)  # TODO: A heuristic will be needed here too.
compute_loopsize() = return LOOPSIZE
determine_optdim() = get_ndims() # NOTE: in @parallel_indices kernels, this could be determined from the indices, but not in the @parallel calls (the heuristic must be the same...): determine_optdim(indices) = return (isa(indices,Expr) ? length(indices.args) : 1)


## FUNCTIONS TO COMPUTE NTHREADS, NBLOCKS

function compute_nthreads_loopopt(maxsize, optdim, halosize) # This is a heuristic, which results typcially in (32,4,1) threads for a 3-D case.
    nthreads = ParallelKernel.compute_nthreads(maxsize; nthreads_max=NTHREADS_MAX_LOOPOPT, flatdim=optdim)
    if (prod(nthreads) < sum(halosize .* nthreads) + 4*prod(max.(halosize,1))) @ArgumentError("@parallel <kernelcall>: the automatic determination of nthreads is not possible for this case. Please specify `nthreads` and `nblocks`.")  end # NOTE: this is a simple heuristic to compute compare the number of threads to the number of halo cells in a 3-D scenario (4*prod(halosize) is to compute the amount of cells in the halo corners). For a 2-D or 1-D scenario this will give alwayse false and raise the error.
    nthreads = (nthreads[1], 4, 1) #TODO: keep? 
    return nthreads
end


## FUNCTIONS TO DEAL WITH MASKS (@WITHIN) AND INDICES

function check_mask_macro(caller::Module)
    if !isdefined(caller, Symbol("@within")) @MethodPluginError("the macro @within is not defined in the caller. You need to load one of the submodules ParallelStencil.FiniteDifferences{1|2|3}D (or a compatible custom module or set of macros).") end
    methods_str = string(methods(getfield(caller, Symbol("@within"))))
    if !occursin(r"(var\"@within\"|@within)\(__source__::LineNumberNode, __module__::Module, .*::String, .*::Symbol\)", methods_str) @MethodPluginError("the signature of the macro @within is not compatible with ParallelStencil (detected signature: \"$methods_str\"). The signature must correspond to the description in ParallelStencil.WITHIN_DOC. See in ParallelStencil.FiniteDifferences{1|2|3}D for examples.") end
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
