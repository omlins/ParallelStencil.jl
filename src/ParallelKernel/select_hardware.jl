##
const SELECT_HARDWARE_DOC = """
    select_hardware(hardware)

Set the runtime hardware architecture used by ParallelKernel backends. When a backend that supports multiple architectures — such as KernelAbstractions — is active, the function records the chosen `hardware` symbol so kernel launch and allocation macros can dispatch to the matching device without re-parsing code. For single-architecture backends the call leaves the preselected hardware unchanged.

# Supported hardware symbols by backend
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
function select_hardware(hardware::Symbol)
end

##
const CURRENT_HARDWARE_DOC = """
    current_hardware()

Return the symbol representing the hardware architecture currently selected for runtime execution. Before any call to [`select_hardware`](@ref) on multi-architecture backends, the default is `:cpu`; single-architecture backends report their fixed hardware symbol. Kernel launch and allocation macros consult this value when constructing backend-specific calls.

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`select_hardware`](@ref)
"""
@doc CURRENT_HARDWARE_DOC
function current_hardware()
end
