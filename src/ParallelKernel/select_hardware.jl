##
const SELECT_HARDWARE_DOC = """
    select_hardware(caller, hardware; package=get_package(caller))

Set the runtime `hardware` architecture used by the backend of the `caller` module. When a backend that supports multiple architectures — such as KernelAbstractions — is active, the function records the chosen `hardware` symbol so kernel launch and allocation macros can dispatch to the matching device without reparsing code. Single-architecture backends remain fixed to their only target.

!!! note "Preferred interface"
    Prefer [`@select_hardware`](@ref) so the caller module and backend are captured automatically.

# Arguments
- `caller::Module`: the module from which the hardware selection is made (easily provided using `@__MODULE__`).
- `hardware::Symbol`: the symbol representing the hardware architecture to select for runtime execution. Supported hardware symbols by backend are:
        - KernelAbstractions: `:cpu`, `:gpu_cuda`, `:gpu_amd`, `:gpu_metal`, `:gpu_oneapi` (defaults to `:cpu`).
        - Threads: `:cpu`.
        - Polyester: `:cpu`.
        - CUDA: `:gpu_cuda`.
        - AMDGPU: `:gpu_amd`.
        - Metal: `:gpu_metal`.

# Keyword arguments
- `package::Symbol`: backend context used for validation (default: `get_package(caller)`).

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`@select_hardware`](@ref), [`current_hardware`](@ref), [`@current_hardware`](@ref)
"""

const SUPPORTED_HARDWARE = Dict(PKG_THREADS              => (hardware_default(PKG_THREADS),),
                                PKG_POLYESTER            => (hardware_default(PKG_POLYESTER),),
                                PKG_CUDA                 => (hardware_default(PKG_CUDA),),
                                PKG_AMDGPU               => (hardware_default(PKG_AMDGPU),),
                                PKG_METAL                => (hardware_default(PKG_METAL),),
                                PKG_KERNELABSTRACTIONS   => (hardware_default(PKG_KERNELABSTRACTIONS), :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi))

@doc SELECT_HARDWARE_DOC
function select_hardware(caller::Module, hardware::Symbol; package::Symbol=get_package(caller))
    createmeta_PK(caller)
    if (package == PKG_NONE) @ArgumentError("ParallelKernel has not been initialized in module $caller (missing @init_parallel_kernel call).") end
    supported = get(SUPPORTED_HARDWARE, package, nothing)
    if isnothing(supported) @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package). Supported packages are: $(join(SUPPORTED_PACKAGES, ", ")).") end
    if !(hardware in supported)
        @ArgumentError("unsupported hardware selection for package $package (obtained: $hardware). Supported hardware symbols are: $(join(supported, ", ")).")
    end
    set_hardware(caller, hardware)
    return nothing
end

##
const CURRENT_HARDWARE_DOC = """
    current_hardware(caller; package=get_package(caller))

Return the symbol representing the hardware architecture currently selected for runtime execution. For multi-architecture backends, the symbol defaults to `:cpu` until changed; single-architecture backends report their fixed hardware symbol. Kernel launch and allocation macros inject this state query when constructing hardware-specific calls.

!!! note "Preferred interface"
    Prefer [`@current_hardware`](@ref) so the caller module and backend are captured automatically.

# Arguments
- `caller::Module`: the module from which the hardware query is made (easily provided using `@__MODULE__`).

# Keyword arguments
- `package::Symbol`: backend context used for lookup (default: `get_package(caller)`).

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`@current_hardware`](@ref), [`select_hardware`](@ref), [`@select_hardware`](@ref)
"""
@doc CURRENT_HARDWARE_DOC
function current_hardware(caller::Module; package::Symbol=get_package(caller))
    return get_hardware(caller)
end

##
const SELECT_HARDWARE_MACRO_DOC = """
    @select_hardware(hardware)

Select the runtime `hardware` architecture for the caller module. The macro captures the caller and backend at parse time and forwards to [`select_hardware`](@ref).

# Arguments
- `hardware::Symbol`: the symbol representing the hardware architecture to select for runtime execution.

See also: [`select_hardware`](@ref), [`@current_hardware`](@ref)
"""
@doc SELECT_HARDWARE_MACRO_DOC
macro select_hardware(hardware)
    caller = __module__
    package = quote_expr(get_package(caller))
    return esc(:(select_hardware($caller, $hardware; package=$package)))
end

##
const CURRENT_HARDWARE_MACRO_DOC = """
    @current_hardware()

Return the symbol representing the runtime hardware architecture selected for the caller module. The macro captures the caller and backend at parse time and forwards to [`current_hardware`](@ref).

See also: [`current_hardware`](@ref), [`@select_hardware`](@ref)
"""
@doc CURRENT_HARDWARE_MACRO_DOC
macro current_hardware()
    caller = __module__
    package = quote_expr(get_package(caller))
    return esc(:(current_hardware($caller; package=$package)))
end

function handle(hardware::Symbol, package::Symbol)
    if package == PKG_KERNELABSTRACTIONS
        return handle_kernelabstractions(hardware)
    else
        @ArgumentError("hardware handle translation is only supported for multi-architecture backends (obtained: $package).")
    end
end

hardware_handle(caller::Module) = handle(current_hardware(caller), get_package(caller))
