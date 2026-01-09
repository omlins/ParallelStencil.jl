import ParallelStencil
using ParallelStencil.ParallelKernel.Exceptions: @KeywordArgumentError

const PK = ParallelStencil.ParallelKernel

# Helper to map KernelAbstractions allocations to the runtime backend.
function runtime_allocator_symbol(kind::Symbol, hardware::Union{Symbol,Nothing})
    target_package, _, _ = PK.resolve_runtime_backend(PK.PKG_KERNELABSTRACTIONS, hardware)
    suffix = PK.allocator_suffix_for(target_package)
    if suffix === nothing
        @KeywordArgumentError("$(PK.ERRMSG_UNSUPPORTED_PACKAGE) (obtained: $target_package).")
    end
    return PK.allocator_function_symbol(kind, suffix)
end

invoke_runtime_allocator(kind::Symbol, hardware::Union{Symbol,Nothing}, args...) =
    getfield(PK, runtime_allocator_symbol(kind, hardware))(args...)


## RUNTIME ALLOCATOR FUNCTIONS

function PK.zeros_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Number}
    return invoke_runtime_allocator(:zeros, hardware, T, blocklength, args...)
end

function PK.ones_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Number}
    return invoke_runtime_allocator(:ones, hardware, T, blocklength, args...)
end

function PK.rand_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{Number,Enum}}
    return invoke_runtime_allocator(:rand, hardware, T, blocklength, args...)
end

function PK.falses_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Bool}
    return invoke_runtime_allocator(:falses, hardware, T, blocklength, args...)
end

function PK.trues_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Bool}
    return invoke_runtime_allocator(:trues, hardware, T, blocklength, args...)
end

function PK.fill_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{Number,Enum}}
    return invoke_runtime_allocator(:fill, hardware, T, blocklength, args...)
end

function PK.zeros_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray}}
    return invoke_runtime_allocator(:zeros, hardware, T, blocklength, args...)
end

function PK.ones_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray}}
    return invoke_runtime_allocator(:ones, hardware, T, blocklength, args...)
end

function PK.rand_kernelabstractions(::Type{T}, blocklength::Val{B}, dims; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray},B}
    return invoke_runtime_allocator(:rand, hardware, T, blocklength, dims)
end

function PK.rand_kernelabstractions(::Type{T}, blocklength, dims...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray}}
    return PK.rand_kernelabstractions(T, blocklength, dims; hardware=hardware)
end

function PK.falses_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray}}
    return invoke_runtime_allocator(:falses, hardware, T, blocklength, args...)
end

function PK.trues_kernelabstractions(::Type{T}, blocklength, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray}}
    return invoke_runtime_allocator(:trues, hardware, T, blocklength, args...)
end

function PK.fill_kernelabstractions(::Type{T}, blocklength::Val{B}, x, args...; hardware::Union{Symbol,Nothing}=nothing) where {T<:Union{SArray,FieldArray},B}
    return invoke_runtime_allocator(:fill, hardware, T, blocklength, x, args...)
end

function PK.fill_kernelabstractions!(A, x; hardware::Union{Symbol,Nothing}=nothing)
    return invoke_runtime_allocator(:fill!, hardware, A, x)
end
