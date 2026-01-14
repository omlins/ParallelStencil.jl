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
@doc SELECT_HARDWARE_DOC
function select_hardware(caller::Module, hardware::Symbol)
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
end
