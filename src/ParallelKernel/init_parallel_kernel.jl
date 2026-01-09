"""
    @init_parallel_kernel(package, numbertype)
    @init_parallel_kernel(package, numbertype, inbounds=..., padding=...)

Initialize the package ParallelKernel, giving access to its main functionality. Creates a module `Data` in the module where `@init_parallel_kernel` is called from; the module `Data` contains the types `Data.Number`, `Data.Array` and `Data.CellArray` (type `?Data` *after* calling `@init_parallel_kernel` to see the full description of the module). 
When the abstraction-layer backend KernelAbstractions is selected, the concrete runtime hardware is chosen later via [`select_hardware`](@ref) and inspected with [`current_hardware`](@ref); consult the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection) for an end-to-end workflow.

!!! note "Convenience modules"
    `Data` and `TData` modules with hardware-specific array aliases are generated only for single-architecture backends (CUDA, AMDGPU, Metal, Threads, Polyester). The KernelAbstractions backend users trade off the additional convenience modules (and warp level macros) for runtime hardware selection instead.

# Arguments
- `package::Module`: the package used for parallelization (CUDA, AMDGPU, Metal, Threads, Polyester, or KernelAbstractions when deferring the hardware decision to runtime).
- `numbertype::DataType`: the type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (e.g. Float32 or Float64). It is contained in `Data.Number` after @init_parallel_kernel.
- `inbounds::Bool=false`: whether to apply `@inbounds` to the kernels by default (overwritable in each kernel definition).
- `padding::Bool=false`: whether to apply padding to the fields allocated with macros from [`ParallelKernel.FieldAllocators`](@ref).

See also: [`Data`](@ref), [`select_hardware`](@ref), [`current_hardware`](@ref)
"""
macro init_parallel_kernel(args...)
    check_already_initialized(__module__)
    posargs, kwargs_expr = split_args(args)
    if (length(args) > 4)            @ArgumentError("too many arguments.")
    elseif (0 < length(posargs) < 2) @ArgumentError("there must be either two or zero positional arguments.")
    end
    kwargs = split_kwargs(kwargs_expr)
    if (length(posargs) == 2) package, numbertype_val = extract_posargs_init(__module__, posargs...)
    else                      package, numbertype_val = extract_kwargs_init(__module__, kwargs)
    end
    inbounds_val, padding_val = extract_kwargs_nopos(__module__, kwargs)
    if (package == PKG_NONE) @ArgumentError("the package argument cannot be ommited.") end #TODO: this error message will disappear, once the package can be defined at runtime.
    esc(init_parallel_kernel(__module__, package, numbertype_val, inbounds_val, padding_val))
end

function init_parallel_kernel(caller::Module, package::Symbol, numbertype::DataType, inbounds::Bool, padding::Bool; datadoc_call=:(), parent_module::String="ParallelKernel")
    data_module = nothing
    tdata_module = nothing
    indextype = INT_THREADS
    if package == PKG_CUDA
        if (isinteractive() && !is_installed("CUDA")) @NotInstalledError("CUDA was selected as package for parallelization, but CUDA.jl is not installed. CUDA functionality is provided as an extension of $parent_module and CUDA.jl needs therefore to be installed independently (type `add CUDA` in the julia package manager).") end
        indextype          = INT_CUDA
    elseif package == PKG_AMDGPU
        if (isinteractive() && !is_installed("AMDGPU")) @NotInstalledError("AMDGPU was selected as package for parallelization, but AMDGPU.jl is not installed. AMDGPU functionality is provided as an extension of $parent_module and AMDGPU.jl needs therefore to be installed independently (type `add AMDGPU` in the julia package manager).") end
        indextype          = INT_AMDGPU
    elseif package == PKG_KERNELABSTRACTIONS
        if (isinteractive() && !is_installed("KernelAbstractions")) @NotInstalledError("KernelAbstractions was selected as package for parallelization, but KernelAbstractions.jl is not installed. KernelAbstractions functionality is provided as an extension of $parent_module and KernelAbstractions.jl needs therefore to be installed independently (type `add KernelAbstractions` in the julia package manager).") end
        indextype                = INT_KERNELABSTRACTIONS
    elseif package == PKG_METAL
        if (isinteractive() && !is_installed("Metal")) @NotInstalledError("Metal was selected as package for parallelization, but Metal.jl is not installed. Metal functionality is provided as an extension of $parent_module and Metal.jl needs therefore to be installed independently (type `add Metal` in the julia package manager).") end
        indextype          = INT_METAL
    elseif package == PKG_POLYESTER
        if (isinteractive() && !is_installed("Polyester")) @NotInstalledError("Polyester was selected as package for parallelization, but Polyester.jl is not installed. Multi-threading using Polyester is provided as an extension of $parent_module and Polyester.jl needs therefore to be installed independently (type `add Polyester` in the julia package manager).") end
        indextype          = INT_POLYESTER
    elseif package == PKG_THREADS
        indextype          = INT_THREADS
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
    create_convenience_modules = convenience_modules_enabled(package)
    if create_convenience_modules
        data_module  = data_module_for(package, numbertype, indextype)
        tdata_module = tdata_module_for(package)
    end
    pkg_import_cmd = define_import(caller, package, parent_module)
    # TODO: before it was ParallelStencil.ParallelKernel.PKG_THREADS, which activated it all weight i think, which should not be
    ad_init_cmd = :(ParallelStencil.ParallelKernel.AD.init_AD($package))
    @eval(caller, $pkg_import_cmd)
    if create_convenience_modules
        if !isdefined(caller, :Data) || (@eval(caller, isa(Data, Module)) &&  length(symbols(caller, :Data)) == 1)  # Only if the module Data does not exist in the caller or is empty, create it.
            if (datadoc_call==:())
                if (numbertype == NUMBERTYPE_NONE) datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC_NUMBERTYPE_NONE Data) 
                else                               datadoc_call = :(@doc ParallelStencil.ParallelKernel.DATA_DOC Data)
                end
            end
            @eval(caller, $data_module)
            @eval(caller, $datadoc_call)
        elseif isdefined(caller, :Data) && isdefined(caller.Data, :Device)
            if !isinteractive() @warn "Module Data from previous module initialization found in caller module ($caller); module Data not created. Note: this warning is only shown in non-interactive mode." end
        else
            @warn "Module Data cannot be created in caller module ($caller) as there is already a user defined symbol (module/variable...) with this name. ParallelStencil is still usable but without the features of the Data module."
        end
        if !isdefined(caller, :TData) || (@eval(caller, isa(TData, Module)) &&  length(symbols(caller, :TData)) == 1)  # Only if the module TData does not exist in the caller or is empty, create it.
            @eval(caller, $tdata_module)
        elseif isdefined(caller, :TData) && isdefined(caller.TData, :Device)
            if !isinteractive() @warn "Module TData from previous module initialization found in caller module ($caller); module TData not created. Note: this warning is only shown in non-interactive mode." end
        else
            @warn "Module TData cannot be created in caller module ($caller) as there is already a user defined symbol (module/variable...) with this name. ParallelStencil is still usable but without the features of the TData module."
        end             
    else
        # KernelAbstractions relies on runtime hardware selection rather than fixed Data/TData convenience modules.
        if isdefined(caller, :Data) && isdefined(caller.Data, :Device)
            data_module = Data_none()
            @eval(caller, $data_module)
        end
        if isdefined(caller, :TData) && isdefined(caller.TData, :Device)
            tdata_module = TData_none()
            @eval(caller, $tdata_module)
        end
    end
    @eval(caller, $ad_init_cmd)
    set_package(caller, package)
    set_numbertype(caller, numbertype)
    set_inbounds(caller, inbounds)
    set_padding(caller, padding)
    reset_runtime_hardware!(package) # ensure runtime selection matches backend default on init
    set_initialized(caller, true)
    return nothing
end


function Metadata_PK()
    :(module $MOD_METADATA_PK # NOTE: there cannot be any newline before 'module $MOD_METADATA_PK' or it will create a begin end block and the module creation will fail.
        let
            global set_initialized, is_initialized, set_package, get_package, set_numbertype, get_numbertype, set_inbounds, get_inbounds, set_padding, get_padding
            _is_initialized::Bool       = false
            package::Symbol             = $(quote_expr(PKG_NONE))
            numbertype::DataType        = $NUMBERTYPE_NONE
            inbounds::Bool              = $INBOUNDS_DEFAULT
            padding::Bool               = $PADDING_DEFAULT
            set_initialized(flag::Bool) = (_is_initialized = flag)
            is_initialized()            = _is_initialized
            set_package(pkg::Symbol)    = (package = pkg)
            get_package()               = package
            set_numbertype(T::DataType) = (numbertype = T)
            get_numbertype()            = numbertype
            set_inbounds(flag::Bool)    = (inbounds = flag)
            get_inbounds()              = inbounds
            set_padding(flag::Bool)     = (padding = flag)
            get_padding()               = padding
        end
    end)
end

createmeta_PK(caller::Module) = if !hasmeta_PK(caller) @eval(caller, $(Metadata_PK())) end


macro is_initialized() is_initialized(__module__) end
macro get_package() esc(get_package(__module__)) end # NOTE: escaping is required here, to avoid that the symbol is evaluated in this module, instead of just being returned as a symbol.
macro get_numbertype() get_numbertype(__module__) end
macro get_inbounds() get_inbounds(__module__) end
macro get_padding() get_padding(__module__) end
let
    global is_initialized, set_initialized, set_package, get_package, set_numbertype, get_numbertype, set_inbounds, get_inbounds, set_padding, get_padding, check_initialized, check_already_initialized    
    set_initialized(caller::Module, flag::Bool) = (createmeta_PK(caller); @eval(caller, $MOD_METADATA_PK.set_initialized($flag)))
    is_initialized(caller::Module)              = hasmeta_PK(caller) && @eval(caller, $MOD_METADATA_PK.is_initialized())
    set_package(caller::Module, pkg::Symbol)    = (createmeta_PK(caller); @eval(caller, $MOD_METADATA_PK.set_package($(quote_expr(pkg)))))
    get_package(caller::Module)                 = hasmeta_PK(caller) ? @eval(caller, $MOD_METADATA_PK.get_package()) : PKG_NONE
    set_numbertype(caller::Module, T::DataType) = (createmeta_PK(caller); @eval(caller, $MOD_METADATA_PK.set_numbertype($T)))
    get_numbertype(caller::Module)              = hasmeta_PK(caller) ? @eval(caller, $MOD_METADATA_PK.get_numbertype()) : NUMBERTYPE_NONE
    set_inbounds(caller::Module, flag::Bool)    = (createmeta_PK(caller); @eval(caller, $MOD_METADATA_PK.set_inbounds($flag)))
    get_inbounds(caller::Module)                = hasmeta_PK(caller) ? @eval(caller, $MOD_METADATA_PK.get_inbounds()) : INBOUNDS_DEFAULT
    set_padding(caller::Module, flag::Bool)     = (createmeta_PK(caller); @eval(caller, $MOD_METADATA_PK.set_padding($flag)))
    get_padding(caller::Module)                 = hasmeta_PK(caller) ? @eval(caller, $MOD_METADATA_PK.get_padding()) : PADDING_DEFAULT
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
    if (:package in keys(kwargs))    package = kwargs[:package]; check_package(package)
    else                             package = PKG_NONE
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
    if (:padding in keys(kwargs))  padding_val = eval_arg(caller, kwargs[:padding]); check_padding(padding_val)
    else                           padding_val = false
    end
    return inbounds_val, padding_val
end

function define_import(caller::Module, package::Symbol, parent_module::String)
    if package == PKG_THREADS
        return :()
    else 
        error_msg = "package $caller is missing $package in its (weak) dependencies ($package was selected as package for parallelization using $parent_module; $package functionality is provided as an extension of $parent_module and $(package).jl needs therefore to be added as an explicit (weak) dependency)."
        return :(
            try
                import $package
            catch e
                throw(ParallelStencil.ParallelKernel.MissingDependencyError($error_msg))
            end
        )
    end
end
