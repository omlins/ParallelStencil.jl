"""
    @init_parallel_kernel(package, numbertype)

Initialize the package ParallelKernel, giving access to its main functionality. Creates a module `Data` in the module where `@init_parallel_kernel` is called from. The module `Data` contains the types as `Data.Number`, `Data.Array` and `Data.CellArray` (type `?Data` *after* calling `@init_parallel_kernel` to see the full description of the module).

# Arguments
- `package::Module`: the package used for parallelization (CUDA, AMDGPU or Threads).
- `numbertype::DataType`: the type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (e.g. Float32 or Float64). It is contained in `Data.Number` after @init_parallel_kernel.
- `inbounds::Bool=false`: whether to apply `@inbounds` to the kernels by default (overwritable in each kernel definition).

See also: [`Data`](@ref)
"""
macro init_parallel_kernel(args...)
    check_already_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    if (length(args) > 3)            @ArgumentError("too many arguments.")
    elseif (0 < length(posargs) < 2) @ArgumentError("there must be either two or zero positional arguments.")
    end
    kwargs = split_kwargs(kwargs_expr)
    if (length(posargs) == 2) package, numbertype_val = extract_posargs_init(__module__, posargs...)
    else                      package, numbertype_val = extract_kwargs_init(__module__, kwargs)
    end
    inbounds_val = extract_kwargs_nopos(__module__, kwargs)
    if (package == PKG_NONE) @ArgumentError("the package argument cannot be ommited.") end #TODO: this error message will disappear, once the package can be defined at runtime.
    esc(init_parallel_kernel(__module__, package, numbertype_val, inbounds_val))
end

function init_parallel_kernel(caller::Module, package::Symbol, numbertype::DataType, inbounds::Bool; datadoc_call=:())
    if package == PKG_CUDA
        if (!is_installed("CUDA")) @NotInstalledError("CUDA was selected as package for parallelization, but CUDA.jl is not installed. CUDA functionality is provided with an extension of ParallelStencil and CUDA.jl needs therefore to be installed independently.") end
        indextype          = INT_CUDA
        data_module        = Data_cuda(numbertype, indextype)
        data_module_shared = Data_shared(numbertype, indextype)
        pkg_import_cmd     = :(import CUDA)
    elseif package == PKG_AMDGPU
        if (!is_installed("AMDGPU")) @NotInstalledError("AMDGPU was selected as package for parallelization, but AMDGPU.jl is not installed. AMDGPU functionality is provided with an extension of ParallelStencil and AMDGPU.jl needs therefore to be installed independently.") end
        indextype          = INT_AMDGPU
        data_module        = Data_amdgpu(numbertype, indextype)
        data_module_shared = Data_shared(numbertype, indextype)
        pkg_import_cmd     = :(import AMDGPU)
    elseif package == PKG_THREADS
        indextype          = INT_THREADS
        data_module        = Data_threads(numbertype, indextype)
        data_module_shared = Data_shared(numbertype, indextype)
        pkg_import_cmd     = :()
    end
    ad_init_cmd = :(ParallelStencil.ParallelKernel.AD.init_AD(ParallelStencil.ParallelKernel.PKG_THREADS))
    if !isdefined(caller, :Data) || (@eval(caller, isa(Data, Module)) &&  length(symbols(caller, :Data)) == 1)  # Only if the module Data does not exist in the caller or is empty, create it.
        if (datadoc_call==:())
            if (numbertype == NUMBERTYPE_NONE) datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC_NUMBERTYPE_NONE Data) 
            else                               datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC Data)
            end
        end
        @eval(caller, $pkg_import_cmd)
        @eval(caller, $data_module)
        @eval(caller.Data, $data_module_shared)
        @eval(caller, $datadoc_call)
    elseif isdefined(caller, :Data) && isdefined(caller.Data, :DeviceArray)
        if !isinteractive() @warn "Module Data from previous module initialization found in caller module ($caller); module Data not created. Note: this warning is only shown in non-interactive mode." end
    else
        @warn "Module Data cannot be created in caller module ($caller) as there is already a user defined symbol (module/variable...) with this name. ParallelStencil is still usable but without the features of the Data module."
    end
    @eval(caller, $ad_init_cmd)
    set_package(caller, package)
    set_numbertype(caller, numbertype)
    set_inbounds(caller, inbounds)
    set_initialized(caller, true)
    return nothing
end


macro is_initialized() is_initialized(__module__) end
macro get_package() esc(get_package(__module__)) end # NOTE: escaping is required here, to avoid that the symbol is evaluated in this module, instead of just being returned as a symbol.
macro get_numbertype() get_numbertype(__module__) end
macro get_inbounds() get_inbounds(__module__) end
let
    global is_initialized, set_initialized, set_package, get_package, set_numbertype, get_numbertype, set_inbounds, get_inbounds, check_initialized, check_already_initialized
    _is_initialized::Dict{Module, Bool}         = Dict{Module, Bool}()
    package::Dict{Module, Symbol}               = Dict{Module, Symbol}()
    numbertype::Dict{Module, DataType}          = Dict{Module, DataType}()
    inbounds::Dict{Module, Bool}                = Dict{Module, Bool}()
    set_initialized(caller::Module, flag::Bool) = (_is_initialized[caller] = flag)
    is_initialized(caller::Module)              = haskey(_is_initialized, caller) && _is_initialized[caller]
    set_package(caller::Module, pkg::Symbol)    = (package[caller] = pkg)
    get_package(caller::Module)                 =  package[caller]
    set_numbertype(caller::Module, T::DataType) = (numbertype[caller] = T)
    get_numbertype(caller::Module)              =  numbertype[caller]
    set_inbounds(caller::Module, flag::Bool)    = (inbounds[caller] = flag)
    get_inbounds(caller::Module)                =  inbounds[caller]
    check_initialized(caller::Module)           = if !is_initialized(caller) @NotInitializedError("no ParallelKernel macro or function can be called before @init_parallel_kernel in each module (missing call in $caller).") end
    check_already_initialized(caller::Module)   = if is_initialized(caller) @IncoherentCallError("ParallelKernel has already been initialized for the module $caller.") end
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

function extract_kwargs_nopos(caller::Module, kwargs::Dict)
    if (:inbounds in keys(kwargs)) inbounds_val = eval_arg(caller, kwargs[:inbounds]); check_inbounds(inbounds_val)
    else                           inbounds_val = false
    end
    return inbounds_val
end
