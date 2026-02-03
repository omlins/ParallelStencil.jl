##
const SELECT_HARDWARE_DOC = """
    select_hardware(caller, hardware)

Set the runtime `hardware` architecture used by the ParallelKernel backend of the `caller` module. When a backend that supports multiple architectures — such as KernelAbstractions — is active, the function records the chosen `hardware` symbol so kernel launch and allocation macros can dispatch to the matching device without reparsing code. For single-architecture backends the hardware is preset and calling this function is not necessary (but allowed).

# Arguments
- `caller::Module`: the module from which the hardware selection is made (easily provided using `@__MODULE__`).
- `hardware::Symbol`: the symbol representing the hardware architecture to select for runtime execution. Supported hardware symbols by backend are:
        - KernelAbstractions: `:cpu`, `:gpu_cuda`, `:gpu_amd`, `:gpu_metal`, `:gpu_oneapi` (defaults to `:cpu`).
        - Threads: `:cpu`.
        - Polyester: `:cpu`.
        - CUDA: `:gpu_cuda`.
        - AMDGPU: `:gpu_amd`.
        - Metal: `:gpu_metal`.

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`current_hardware`](@ref)
"""

const SUPPORTED_HARDWARE = Dict(PKG_THREADS              => (hardware_default(PKG_THREADS),),
                                PKG_POLYESTER            => (hardware_default(PKG_POLYESTER),),
                                PKG_CUDA                 => (hardware_default(PKG_CUDA),),
                                PKG_AMDGPU               => (hardware_default(PKG_AMDGPU),),
                                PKG_METAL                => (hardware_default(PKG_METAL),),
                                PKG_KERNELABSTRACTIONS   => (hardware_default(PKG_KERNELABSTRACTIONS), :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi))

@doc SELECT_HARDWARE_DOC
function select_hardware(caller::Module, hardware::Symbol)
    createmeta_PK(caller)
    package = get_package(caller)
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
    current_hardware(caller)

Return the symbol representing the hardware architecture currently selected for runtime execution. Before any call to [`select_hardware`](@ref) on multi-architecture backends, the default is `:cpu`; single-architecture backends report their fixed hardware symbol. Kernel launch and allocation macros inject this state query when constructing hardware-specific calls.

# Arguments
- `caller::Module`: the module from which the hardware query is made (easily provided using `@__MODULE__`).

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`select_hardware`](@ref)
"""
@doc CURRENT_HARDWARE_DOC
function current_hardware(caller::Module)
    return get_hardware(caller)
end

function handle(hardware::Symbol)
    package = get_package(@__MODULE__)
    if package == PKG_KERNELABSTRACTIONS
        if hardware == :cpu
            return KernelAbstractions.CPU()
        elseif hardware == :gpu_cuda
            return KernelAbstractions.CUDABackend()
        elseif hardware == :gpu_amd
            return KernelAbstractions.ROCBackend()
        elseif hardware == :gpu_metal
            return KernelAbstractions.MetalBackend()
        elseif hardware == :gpu_oneapi
            return KernelAbstractions.oneAPIBackend()
        end
        @ArgumentError("unsupported KernelAbstractions hardware symbol (obtained: $hardware).")
    else
        @ArgumentError("hardware handle translation is only supported for multi-architecture backends (obtained: $package).")
    end
end

hardware_handle(caller::Module) = handle(current_hardware(caller))
