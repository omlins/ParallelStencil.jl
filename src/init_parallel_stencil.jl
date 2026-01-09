"""
    @init_parallel_stencil(package, numbertype, ndims)
    @init_parallel_stencil(package, numbertype, ndims, inbounds=...)
    @init_parallel_stencil(package=..., ndims=..., inbounds=...)

Initialize the package ParallelStencil, giving access to its main functionality. Creates a module `Data` in the module where `@init_parallel_stencil` is called from; the module `Data` contains the types `Data.Number`, `Data.Array` and `Data.CellArray` (type `?Data` *after* calling `@init_parallel_stencil` to see the full description of the module).
When the abstraction-layer backend KernelAbstractions is selected, the concrete runtime hardware is chosen later via [`select_hardware`](@ref) and inspected with [`current_hardware`](@ref); consult the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection) for an end-to-end workflow.

!!! note "Convenience modules"
    `Data` and `TData` modules with hardware-specific array aliases are generated only for single-architecture backends (CUDA, AMDGPU, Metal, Threads, Polyester). KernelAbstractions users trade off the additional convenience modules for runtime hardware selection instead.

# Arguments
- `package::Module`: the package used for parallelization (CUDA, AMDGPU, Metal, Threads, Polyester, or KernelAbstractions when deferring the hardware decision to runtime).
- `numbertype::DataType`: the type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (e.g. Float32 or Float64). It is contained in `Data.Number` after @init_parallel_stencil. The `numbertype` can be omitted if the other arguments are given as keyword arguments (in that case, the `numbertype` will have to be given explicitly when using the types provided by the module `Data`).
- `ndims::Integer`: the number of dimensions used for the stencil computations in the kernels: 1, 2 or 3 (overwritable in each kernel definition).
- `inbounds::Bool=false`: whether to apply `@inbounds` to the kernels by default (overwritable in each kernel definition).

See also: [`Data`](@ref), [`select_hardware`](@ref), [`current_hardware`](@ref)
"""
macro init_parallel_stencil(args...)
    posargs, kwargs_expr = split_args(args)
    if (length(args) > 6)            @ArgumentError("too many arguments.")
    elseif (0 < length(posargs) < 3) @ArgumentError("there must be either three or zero positional arguments.")
    end
    kwargs = split_kwargs(kwargs_expr)
    if (length(posargs) == 3) package, numbertype_val, ndims_val = extract_posargs_init(__module__, posargs...)
    else                      package, numbertype_val, ndims_val = extract_kwargs_init(__module__, kwargs)
    end
    inbounds_val, padding_val, memopt_val, nonconst_metadata_val = extract_kwargs_nopos(__module__, kwargs)
    if (package == PKG_NONE) @ArgumentError("the package argument cannot be ommited.") end #TODO: this error message will disappear, once the package can be defined at runtime.
    if (package == PKG_POLYESTER && padding_val) @ArgumentError("padding is not yet supported for Polyester.") end
    check_already_initialized(__module__, package, numbertype_val, ndims_val, inbounds_val, padding_val, memopt_val, nonconst_metadata_val)
    esc(init_parallel_stencil(__module__, package, numbertype_val, ndims_val, inbounds_val, padding_val, memopt_val, nonconst_metadata_val))
end

function init_parallel_stencil(caller::Module, package::Symbol, numbertype::DataType, ndims::Integer, inbounds::Bool, padding::Bool, memopt::Bool, nonconst_metadata::Bool)
    if package == PKG_KERNELABSTRACTIONS
        datadoc_call = :()
    elseif numbertype == NUMBERTYPE_NONE
        datadoc_call = :(@doc replace(ParallelStencil.ParallelKernel.DATA_DOC_NUMBERTYPE_NONE, "ParallelKernel" => "ParallelStencil", "@init_parallel_kernel" => "@init_parallel_stencil") Data)
    else
        datadoc_call = :(@doc replace(ParallelStencil.ParallelKernel.DATA_DOC,                 "ParallelKernel" => "ParallelStencil", "@init_parallel_kernel" => "@init_parallel_stencil") Data)
    end
    return_expr = ParallelKernel.init_parallel_kernel(caller, package, numbertype, inbounds, padding; datadoc_call=datadoc_call, parent_module="ParallelStencil")
    set_package(caller, package)
    set_numbertype(caller, numbertype)
    set_ndims(caller, ndims)
    set_inbounds(caller, inbounds)
    set_padding(caller, padding)
    set_memopt(caller, memopt)
    set_nonconst_metadata(caller, nonconst_metadata)
    if package == PKG_KERNELABSTRACTIONS
        ParallelKernel.reset_runtime_hardware!(package) # ensure runtime selection wrappers reflect the default KernelAbstractions target
    end
    set_initialized(caller, true)
    return return_expr
end


function Metadata_PS()
    :(module $MOD_METADATA_PS # NOTE: there cannot be any newline before 'module $MOD_METADATA_PS' or it will create a begin end block and the module creation will fail.
        let
            global set_initialized, is_initialized, set_package, get_package, set_numbertype, get_numbertype, set_ndims, get_ndims, set_inbounds, get_inbounds, set_padding, get_padding, set_memopt, get_memopt, set_nonconst_metadata, get_nonconst_metadata
            _is_initialized::Bool             = false
            package::Symbol                   = $(quote_expr(PKG_NONE))
            numbertype::DataType              = $NUMBERTYPE_NONE
            ndims::Integer                    = $NDIMS_NONE
            inbounds::Bool                    = $INBOUNDS_DEFAULT
            padding::Bool                     = $PADDING_DEFAULT
            memopt::Bool                      = $MEMOPT_DEFAULT
            nonconst_metadata::Bool           = $NONCONST_METADATA_DEFAULT
            set_initialized(flag::Bool)       = (_is_initialized = flag)
            is_initialized()                  = _is_initialized
            set_package(pkg::Symbol)          = (package = pkg)
            get_package()                     = package
            set_numbertype(T::DataType)       = (numbertype = T)
            get_numbertype()                  = numbertype
            set_ndims(n::Integer)             = (ndims = n)
            get_ndims()                       = ndims
            set_inbounds(flag::Bool)          = (inbounds = flag)
            get_inbounds()                    = inbounds
            set_padding(flag::Bool)           = (padding = flag)
            get_padding()                     = padding
            set_memopt(flag::Bool)            = (memopt = flag)
            get_memopt()                      = memopt
            set_nonconst_metadata(flag::Bool) = (nonconst_metadata = flag)
            get_nonconst_metadata()           = nonconst_metadata
        end
    end)
end

createmeta_PS(caller::Module) = if !hasmeta_PS(caller) @eval(caller, $(Metadata_PS())) end


macro is_initialized() is_initialized(__module__) end
macro get_package() esc(get_package(__module__)) end # NOTE: escaping is required here, to avoid that the symbol is evaluated in this module, instead of just being returned as a symbol.
macro get_numbertype() get_numbertype(__module__) end
macro get_ndims() get_ndims(__module__) end
macro get_inbounds() get_inbounds(__module__) end
macro get_padding() get_padding(__module__) end
macro get_memopt() get_memopt(__module__) end
macro get_nonconst_metadata() get_nonconst_metadata(__module__) end
let
    global is_initialized, set_initialized, set_package, get_package, set_numbertype, get_numbertype, set_ndims, get_ndims, set_inbounds, get_inbounds, set_padding, get_padding, set_memopt, get_memopt, set_nonconst_metadata, get_nonconst_metadata, check_initialized, check_already_initialized
    set_initialized(caller::Module, flag::Bool)       = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_initialized($flag)))
    is_initialized(caller::Module)                    = hasmeta_PS(caller) && @eval(caller, $MOD_METADATA_PS.is_initialized())
    set_package(caller::Module, pkg::Symbol)          = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_package($(quote_expr(pkg)))))
    get_package(caller::Module)                       = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_package()) : PKG_NONE
    set_numbertype(caller::Module, T::DataType)       = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_numbertype($T)))
    get_numbertype(caller::Module)                    = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_numbertype()) : NUMBERTYPE_NONE
    set_ndims(caller::Module, n::Integer)             = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_ndims($n)))
    get_ndims(caller::Module)                         = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_ndims()) : NDIMS_NONE
    set_inbounds(caller::Module, flag::Bool)          = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_inbounds($flag)))
    get_inbounds(caller::Module)                      = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_inbounds()) : INBOUNDS_DEFAULT
    set_padding(caller::Module, flag::Bool)           = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_padding($flag)))
    get_padding(caller::Module)                       = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_padding()) : PADDING_DEFAULT
    set_memopt(caller::Module, flag::Bool)            = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_memopt($flag)))
    get_memopt(caller::Module)                        = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_memopt()) : MEMOPT_DEFAULT
    set_nonconst_metadata(caller::Module, flag::Bool) = (createmeta_PS(caller); @eval(caller, $MOD_METADATA_PS.set_nonconst_metadata($flag)))
    get_nonconst_metadata(caller::Module)             = hasmeta_PS(caller) ? @eval(caller, $MOD_METADATA_PS.get_nonconst_metadata()) : NONCONST_METADATA_DEFAULT
    check_initialized(caller::Module)                 = if !is_initialized(caller) @NotInitializedError("no ParallelStencil macro or function can be called before @init_parallel_stencil in each module (missing call in $caller).") end

    function check_already_initialized(caller::Module, package::Symbol, numbertype::DataType, ndims::Integer, inbounds::Bool, padding::Bool, memopt::Bool, nonconst_metadata::Bool)
        if is_initialized(caller)
            if package==get_package(caller) && numbertype==get_numbertype(caller) && ndims==get_ndims(caller) && inbounds==get_inbounds(caller) && padding==get_padding(caller) && memopt==get_memopt(caller) && nonconst_metadata==get_nonconst_metadata(caller)
                if !isinteractive() @warn "ParallelStencil has already been initialized for the module $caller, with the same arguments. You are likely using ParallelStencil in an inconsistent way: @init_parallel_stencil should only be called once at the beginning of each module, right after 'using ParallelStencil'. Note: this warning is only shown in non-interactive mode." end
            else
                @IncoherentCallError("ParallelStencil has already been initialized for the module $caller, with different arguments. If you are using ParallelStencil interactively in the REPL and want to avoid restarting Julia, then you can call ParallelStencil.@reset_parallel_stencil() and rerun all parts of your code (in module $caller) that use ParallelStencil features (including kernel definitions and array allocations). If you are using ParallelStencil non-interactively, then you are using ParallelStencil in an invalid way: @init_parallel_stencil should only be called once at the beginning of each module, right after 'using ParallelStencil'.")
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
    else                        ndims_val = NDIMS_NONE
    end
    return package, numbertype_val, ndims_val
end

function extract_kwargs_nopos(caller::Module, kwargs::Dict)
    inbounds_val, padding_val = ParallelKernel.extract_kwargs_nopos(caller, kwargs)
    if (:memopt in keys(kwargs)) memopt_val = eval_arg(caller, kwargs[:memopt]); check_memopt(memopt_val)
    else                         memopt_val = false
    end
    if (:nonconst_metadata in keys(kwargs)) nonconst_metadata_val = eval_arg(caller, kwargs[:nonconst_metadata]); check_nonconst_metadata(nonconst_metadata_val)
    else                                    nonconst_metadata_val = false
    end
    return inbounds_val, padding_val, memopt_val, nonconst_metadata_val
end