##
const SELECT_HARDWARE_DOC = """
    select_hardware(hardware)

Set the runtime hardware architecture used by ParallelKernel backends. When a backend that supports multiple architectures — such as KernelAbstractions — is active, the function records the chosen `hardware` symbol so kernel launch and allocation macros can dispatch to the matching device without reparsing code. For single-architecture backends the call leaves the preselected hardware unchanged.

# Arguments
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
const RUNTIME_HARDWARE_OPTIONS = Dict(
    PKG_KERNELABSTRACTIONS => (:cpu, :gpu_cuda, :gpu_amd, :gpu_metal, :gpu_oneapi),
    PKG_THREADS => (:cpu,),
    PKG_POLYESTER => (:cpu,),
    PKG_CUDA => (:gpu_cuda,),
    PKG_AMDGPU => (:gpu_amd,),
    PKG_METAL => (:gpu_metal,)
)

##
const CURRENT_HARDWARE_DOC = """
    current_hardware()

Return the symbol representing the hardware architecture currently selected for runtime execution. Before any call to [`select_hardware`](@ref) on multi-architecture backends, the default is `:cpu`; single-architecture backends report their fixed hardware symbol. Kernel launch and allocation macros consult this value when constructing hardware-specific calls.

For workflow guidance refer to the [interactive prototyping runtime selection section](@ref interactive-prototyping-runtime-hardware-selection).

See also: [`select_hardware`](@ref)
"""
let hardware = default_hardware_for(PKG_THREADS)
    hardware_options(pkg::Symbol) = get(RUNTIME_HARDWARE_OPTIONS, pkg, nothing)

    function validate_hardware_symbol(pkg::Symbol, target::Symbol)
        options = hardware_options(pkg)
        if options === nothing
            @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $pkg).")
        end
        if !(target in options)
            @ArgumentError("unsupported hardware symbol $(target) for package $(pkg). Supported symbols: $(join(string.(options), ", ")).")
        end
        return options
    end

    function select_kernelabstractions_handle(target::Symbol)
        ka = Base.require(:KernelAbstractions)
        if target == :cpu
            return ka.CPU()
        elseif target == :gpu_cuda
            return ka.CUDABackend()
        elseif target == :gpu_amd
            return ka.ROCBackend()
        elseif target == :gpu_metal
            return ka.MetalBackend()
        elseif target == :gpu_oneapi
            return ka.oneAPIBackend()
        else
            @ArgumentError("unsupported hardware symbol $(target) for KernelAbstractions.")
        end
    end

    @doc SELECT_HARDWARE_DOC
    function select_hardware(target::Symbol)
        check_initialized(@__MODULE__)
        package = get_package(@__MODULE__)
        options = validate_hardware_symbol(package, target)
        if package == PKG_KERNELABSTRACTIONS
            hardware = target
            return select_kernelabstractions_handle(target)
        else
            hardware = options[1]
            return hardware
        end
    end

    @doc CURRENT_HARDWARE_DOC
    function current_hardware()
        return hardware
    end

    function handle(target::Symbol)
        check_initialized(@__MODULE__)
        package = get_package(@__MODULE__)
        validate_hardware_symbol(package, target)
        if package != PKG_KERNELABSTRACTIONS
            @ArgumentError("runtime hardware handles are only available for KernelAbstractions.")
        end
        return select_kernelabstractions_handle(target)
    end

    function reset_runtime_hardware!(pkg::Symbol)
        hardware = pkg == PKG_NONE ? default_hardware_for(PKG_THREADS) : default_hardware_for(pkg)
        return hardware
    end
end

