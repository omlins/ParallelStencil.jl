@doc replace(ParallelKernel.HIDE_COMMUNICATION_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro hide_communication(args...) check_initialized(); esc(:(ParallelStencil.ParallelKernel.@hide_communication($(args...)))); end
@doc replace(ParallelKernel.ZEROS_DOC,              "@init_parallel_kernel" => "@init_parallel_stencil") macro zeros(args...)              check_initialized(); esc(:(ParallelStencil.ParallelKernel.@zeros($(args...)))); end
@doc replace(ParallelKernel.ONES_DOC,               "@init_parallel_kernel" => "@init_parallel_stencil") macro ones(args...)               check_initialized(); esc(:(ParallelStencil.ParallelKernel.@ones($(args...)))); end
@doc replace(ParallelKernel.RAND_DOC,               "@init_parallel_kernel" => "@init_parallel_stencil") macro rand(args...)               check_initialized(); esc(:(ParallelStencil.ParallelKernel.@rand($(args...)))); end
@doc replace(ParallelKernel.PARALLEL_INDICES_DOC,   "@init_parallel_kernel" => "@init_parallel_stencil") macro parallel_indices(args...)   check_initialized(); esc(:(ParallelStencil.ParallelKernel.@parallel_indices($(args...)))); end
@doc replace(ParallelKernel.PARALLEL_ASYNC_DOC,     "@init_parallel_kernel" => "@init_parallel_stencil") macro parallel_async(args...)     check_initialized(); esc(:(ParallelStencil.ParallelKernel.@parallel_async($(args...)))); end
@doc replace(ParallelKernel.SYNCHRONIZE_DOC,        "@init_parallel_kernel" => "@init_parallel_stencil") macro synchronize(args...)        check_initialized(); esc(:(ParallelStencil.ParallelKernel.@synchronize($(args...)))); end
# NOTE: @parallel does not appear here as it is extended and therefore defined in parallel.jl
@doc replace(ParallelKernel.GRIDDIM_DOC,            "@init_parallel_kernel" => "@init_parallel_stencil") macro gridDim(args...)            check_initialized(); esc(:(ParallelStencil.ParallelKernel.@gridDim($(args...)))); end
@doc replace(ParallelKernel.BLOCKIDX_DOC,           "@init_parallel_kernel" => "@init_parallel_stencil") macro blockIdx(args...)           check_initialized(); esc(:(ParallelStencil.ParallelKernel.@blockIdx($(args...)))); end
@doc replace(ParallelKernel.BLOCKDIM_DOC,           "@init_parallel_kernel" => "@init_parallel_stencil") macro blockDim(args...)           check_initialized(); esc(:(ParallelStencil.ParallelKernel.@blockDim($(args...)))); end
@doc replace(ParallelKernel.THREADIDX_DOC,          "@init_parallel_kernel" => "@init_parallel_stencil") macro threadIdx(args...)          check_initialized(); esc(:(ParallelStencil.ParallelKernel.@threadIdx($(args...)))); end
@doc replace(ParallelKernel.SYNCTHREADS_DOC,        "@init_parallel_kernel" => "@init_parallel_stencil") macro sync_threads(args...)       check_initialized(); esc(:(ParallelStencil.ParallelKernel.@sync_threads($(args...)))); end
@doc replace(ParallelKernel.SHAREDMEM_DOC,          "@init_parallel_kernel" => "@init_parallel_stencil") macro sharedMem(args...)          check_initialized(); esc(:(ParallelStencil.ParallelKernel.@sharedMem($(args...)))); end
@doc replace(replace(ParallelKernel.PKSHOW_DOC,     "@init_parallel_kernel" => "@init_parallel_stencil"), "pk_show"    => "ps_show")    macro ps_show(args...)     check_initialized(); esc(:(ParallelStencil.ParallelKernel.@pk_show($(args...)))); end
@doc replace(replace(ParallelKernel.PKPRINTLN_DOC,  "@init_parallel_kernel" => "@init_parallel_stencil"), "pk_println" => "ps_println") macro ps_println(args...)  check_initialized(); esc(:(ParallelStencil.ParallelKernel.@pk_println($(args...)))); end


"""
    @init_parallel_stencil(package, numbertype, ndims)

Initialize the package ParallelStencil, giving access to its main functionality. Creates a module `Data` in the module where `@init_parallel_stencil` is called from. The module `Data` contains the types `Data.Number`, `Data.Array` and `Data.DeviceArray` (type `?Data` *after* calling `@init_parallel_stencil` to see the full description of the module).

# Arguments
- `package::Module`: the package used for parallelization (CUDA or Threads).
- `numbertype::DataType`: the type of numbers used by @zeros, @ones and @rand and in all array types of module `Data` (e.g. Float32 or Float64). It is contained in `Data.Number` after @init_parallel_stencil.
- `ndims::Integer`: the number of dimensions used for the stencil computations in the kernels (1, 2 or 3).

See also: [`Data`](@ref)
"""
macro init_parallel_stencil(args...)
    posargs, kwargs_expr = split_args(args)
    if (length(args) > 3)            @ArgumentError("too many arguments.")
    elseif (0 < length(posargs) < 3) @ArgumentError("there must be either three or zero positional arguments.")
    end
    kwargs = split_kwargs(kwargs_expr)
    if (length(posargs) == 3) package, numbertype_val, ndims_val = extract_posargs_init(__module__, posargs...)
    else                      package, numbertype_val, ndims_val = extract_kwargs_init(__module__, kwargs)
    end
    if (package == PKG_NONE) @ArgumentError("the package argument cannot be ommited.") end #TODO: this error message will disappear, once the package can be defined at runtime.
    if (ndims == NDIMS_NONE) @ArgumentError("the ndims argument cannot be ommited.") end #TODO: this error message will disappear, once the ndims can be defined at runtime.
    check_already_initialized(package, numbertype_val, ndims_val)
    esc(init_parallel_stencil(__module__, package, numbertype_val, ndims_val))
end

function init_parallel_stencil(caller::Module, package::Symbol, numbertype::DataType, ndims::Integer)
    datadoc_call = :(@doc replace(ParallelStencil.ParallelKernel.DATA_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") Data)
    ParallelKernel.init_parallel_kernel(caller, package, numbertype; datadoc_call=datadoc_call)
    set_package(package)
    set_numbertype(numbertype)
    set_ndims(ndims)
    set_initialized(true)
    return nothing
end


macro is_initialized() is_initialized() end
macro get_package() get_package() end
macro get_numbertype() get_numbertype() end
macro get_ndims() get_ndims() end
let
    global is_initialized, set_initialized, set_package, get_package, set_numbertype, get_numbertype, set_ndims, get_ndims, check_initialized, check_already_initialized
    _is_initialized::Bool       = false
    package::Symbol             = PKG_NONE
    numbertype::DataType        = NUMBERTYPE_NONE
    ndims::Integer              = NDIMS_NONE
    set_initialized(flag::Bool) = (_is_initialized = flag)
    is_initialized()            = _is_initialized
    set_package(pkg::Symbol)    = (package = pkg)
    get_package()               = package
    set_numbertype(T::DataType) = (numbertype = T)
    get_numbertype()            = numbertype
    set_ndims(n::Integer)       = (ndims = n)
    get_ndims()                 = ndims
    check_initialized()         = if !is_initialized() @NotInitializedError("no macro or function of the module can be called before @init_parallel_stencil.") end

    function check_already_initialized(package::Symbol, numbertype::DataType, ndims::Integer)
        if is_initialized()
            if package==get_package() && numbertype==get_numbertype() && ndims==get_ndims()
                @warn "ParallelStencil has already been initialized, with the same arguments. If you are using ParallelStencil interactively in the REPL, then you can ignore this message. If you are using ParallelStencil non-interactively, then you are likely using ParallelStencil in an inconsistent way: @init_parallel_stencil should only be called once, right after 'using ParallelStencil'."
            else
                @IncoherentCallError("ParallelStencil has already been initialized, with different arguments. If you are using ParallelStencil interactively in the REPL and want to avoid restarting Julia, then you can call ParallelStencil.@reset_parallel_stencil() and rerun all parts of your code that use ParallelStencil features (including kernel definitions and array allocations). If you are using ParallelStencil non-interactively, then you are using ParallelStencil in an invalid way: @init_parallel_stencil should only be called once, right after 'using ParallelStencil'.")
            end
        end
    end
end

function extract_posargs_init(caller::Module, package, numbertype, ndims) # NOTE: this function takes not only symbols: numbertype can be anything that evaluates to a type in the caller and for package will be checked wether it is a symbol in check_package and a proper error message given if not.
    package, numbertype_val = extract_posargs_init(caller, package, numbertype)
    ndims_val = eval_arg(caller, ndims)
    check_ndims(ndims_val)
    return package, numbertype_val, ndims_val
end

function extract_kwargs_init(caller::Module, kwargs::Dict)
    package, numbertype_val = ParallelKernel.extract_kwargs_init(caller, kwargs)
    if (:ndims in keys(kwargs)) ndims_val = eval_arg(caller, kwargs[:ndims]); check_ndims(ndims_val)
    else                        ndims_val = NUMBERTYPE_NONE
    end
    return package, numbertype_val, ndims_val
end
