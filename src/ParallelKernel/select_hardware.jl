const SELECT_HARDWARE_DOC = """
    select_hardware(hardware)

Update the runtime hardware selection for the backend chosen during [`@init_parallel_kernel`](@ref). Use it in interactive workflows to swap the target architecture without redefining kernels.

# Arguments
- `hardware::Symbol`: Backend-specific selector accepted by the initialized backend:
    - KernelAbstractions: `:cpu`, `:gpu_cuda`, `:gpu_amd`, `:gpu_metal`, `:gpu_oneapi`.
    - Threads: `:cpu`.
    - Polyester: `:cpu`.
    - CUDA: `:gpu_cuda`.
    - AMDGPU: `:gpu_amd`.
    - Metal: `:gpu_metal`.

Multi-architecture backends default to `:cpu` immediately after initialization; call this function whenever you need to execute kernels on a different supported architecture. The active selection is reflected by [`current_hardware`](@ref) and is consumed automatically by launch and allocation helpers.

See also: [`current_hardware`](@ref), [`ParallelStencil.select_hardware`](@ref), [Interactive prototyping with runtime hardware selection](@ref interactive-prototyping-with-runtime-hardware-selection)
"""
@doc SELECT_HARDWARE_DOC
function select_hardware(::Symbol)
end

const CURRENT_HARDWARE_DOC = """
    current_hardware()

Return the hardware symbol currently used by ParallelKernel to launch kernels and allocate arrays for the backend chosen during [`@init_parallel_kernel`](@ref). For multi-architecture backends the value starts as `:cpu` until [`select_hardware`](@ref) sets another supported symbol.

See also: [`select_hardware`](@ref), [`ParallelStencil.current_hardware`](@ref), [Interactive prototyping with runtime hardware selection](@ref interactive-prototyping-with-runtime-hardware-selection)
"""
@doc CURRENT_HARDWARE_DOC
function current_hardware()
end
