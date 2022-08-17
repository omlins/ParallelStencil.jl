"""
    @init_parallel_kernel(package, numbertype)

Initialize the package ParallelKernel, giving access to its main functionality. Creates a module `Data` in the module where `@init_parallel_kernel` is called from. The module `Data` contains the types as `Data.Number`, `Data.Array` and `Data.CellArray` (type `?Data` *after* calling `@init_parallel_kernel` to see the full description of the module).

# Arguments
- `package::Module`: the package used for parallelization (CUDA or Threads).
- `numbertype::DataType`: the type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (e.g. Float32 or Float64). It is contained in `Data.Number` after @init_parallel_stencil.

See also: [`Data`](@ref)
"""
macro init_parallel_kernel(args...)
    check_already_initialized()
    posargs, kwargs_expr = split_args(args)
    if (length(args) > 2)            @ArgumentError("too many arguments.")
    elseif (0 < length(posargs) < 2) @ArgumentError("there must be either two or zero positional arguments.")
    end
    kwargs = split_kwargs(kwargs_expr)
    if (length(posargs) == 2) package, numbertype_val = extract_posargs_init(__module__, posargs...)
    else                      package, numbertype_val = extract_kwargs_init(__module__, kwargs)
    end
    if (package == PKG_NONE) @ArgumentError("the package argument cannot be ommited.") end #TODO: this error message will disappear, once the package can be defined at runtime.
    esc(init_parallel_kernel(__module__, package, numbertype_val))
end

function init_parallel_kernel(caller::Module, package::Symbol, numbertype::DataType; datadoc_call=:())
    if package == PKG_CUDA
        data_module = Data_cuda(numbertype)
        import_cmd  = :(import CUDA)
    elseif package == PKG_THREADS
        data_module = Data_threads(numbertype)
        import_cmd  = :()
    end
    if !isdefined(caller, :Data) || (@eval(caller, isa(Data, Module)) &&  length(symbols(caller, :Data)) == 1)  # Only if the module Data does not exist in the caller or is empty, create it.
        if (datadoc_call==:())
            if (numbertype == NUMBERTYPE_NONE) datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC_NUMBERTYPE_NONE Data) 
            else                               datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC Data)
            end
        end
        @eval(caller, $data_module)
        @eval(caller, $datadoc_call)
    elseif isdefined(caller, :Data) && isdefined(caller.Data, :DeviceArray)
        @warn "Module Data from previous module initialization found in caller module ($caller); module Data not created. If you are working interactively in the REPL, then you can ignore this message."
    else
        @warn "Module Data cannot be created in caller module ($caller) as there is already a user defined symbol (module/variable...) with this name. ParallelStencil is still usable but without the features of the Data module."
    end
    @eval(caller, $import_cmd)
    set_package(package)
    set_numbertype(numbertype)
    set_initialized(true)
    return nothing
end


macro is_initialized() is_initialized() end
macro get_package() get_package() end
macro get_numbertype() get_numbertype() end
let
    global is_initialized, set_initialized, set_package, get_package, set_numbertype, get_numbertype, check_initialized, check_already_initialized
    _is_initialized::Bool       = false
    package::Symbol             = PKG_NONE
    numbertype::DataType        = NUMBERTYPE_NONE
    set_initialized(flag::Bool) = (_is_initialized = flag)
    is_initialized()            = _is_initialized
    set_package(pkg::Symbol)    = (package = pkg)
    get_package()               = package
    set_numbertype(T::DataType) = (numbertype = T)
    get_numbertype()            = numbertype
    check_initialized()         = if !is_initialized() @NotInitializedError("no macro or function of the module can be called before @init_parallel_kernel.") end
    check_already_initialized() = if is_initialized() @IncoherentCallError("ParallelKernel has already been initialized.") end
end

function extract_posargs_init(caller::Module, package, numbertype)  # NOTE: this function takes not only symbols: numbertype can be anything that evaluates to a type in the caller and for package will be checked wether it is a symbol in check_package and a proper error message given if not.
    numbertype_val = eval_arg(caller, numbertype)
    check_package(package)
    check_numbertype(numbertype_val)
    return package, numbertype_val
end

function extract_kwargs_init(caller::Module, kwargs::Dict)
    if (:package in keys(kwargs)) package = kwargs[:package]; check_package(package)
    else                          package = PKG_NONE
    end
    if (:numbertype in keys(kwargs)) numbertype_val = eval_arg(caller, kwargs[:numbertype]); check_numbertype(numbertype_val)
    else                             numbertype_val = NUMBERTYPE_NONE
    end
    return package, numbertype_val
end
