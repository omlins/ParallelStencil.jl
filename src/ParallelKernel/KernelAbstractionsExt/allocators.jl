## RUNTIME ALLOCATOR FUNCTIONS

ParallelStencil.ParallelKernel.zeros_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Number}             = (check_datatype_kernelabstractions(T); KernelAbstractions.zeros(backend, T, args...))
ParallelStencil.ParallelKernel.ones_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Number}              = (check_datatype_kernelabstractions(T); KernelAbstractions.ones(backend, T, args...))
ParallelStencil.ParallelKernel.rand_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}  = (check_datatype_kernelabstractions(T, Bool, Enum); KernelAbstractions.rand(backend, T, args...))
ParallelStencil.ParallelKernel.falses_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Bool}              = KernelAbstractions.fill(backend, false, args...)
ParallelStencil.ParallelKernel.trues_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Bool}               = KernelAbstractions.fill(backend, true, args...)
ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}  = KernelAbstractions.fill(backend, convert(T, args[1]), args[2:end]...)

ParallelStencil.ParallelKernel.zeros_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}} = (check_datatype_kernelabstractions(T); ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, T, blocklength, 0, args...))
ParallelStencil.ParallelKernel.ones_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}  = (check_datatype_kernelabstractions(T); ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, T, blocklength, 1, args...))
ParallelStencil.ParallelKernel.rand_kernelabstractions(backend, ::Type{T}, ::Val{B}, dims) where {T<:Union{SArray,FieldArray}, B} = begin
    check_datatype_kernelabstractions(T, Bool, Enum)
    blocklen = (B == 0) ? prod(dims) : B
    storage = KernelAbstractions.rand(backend, eltype(T), blocklen, prod(size(T)), ceil(Int, prod(dims) / blocklen))
    CellArray{T,length(dims),B, typeof(storage)}(storage, dims)
end
ParallelStencil.ParallelKernel.rand_kernelabstractions(backend, ::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}} = ParallelStencil.ParallelKernel.rand_kernelabstractions(backend, T, blocklength, dims)
ParallelStencil.ParallelKernel.falses_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}} = ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, T, blocklength, false, args...)
ParallelStencil.ParallelKernel.trues_kernelabstractions(backend, ::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}  = ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, T, blocklength, true, args...)

function ParallelStencil.ParallelKernel.fill_kernelabstractions(backend, ::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype_kernelabstractions(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    blocklen = (B == 0) ? prod(size(T)) : B
    storage = KernelAbstractions.zeros(backend, eltype(T), blocklen, prod(size(T)), ceil(Int, prod(args) / blocklen))
    return CellArrays.fill!(CellArray{T,length(args),B, typeof(storage)}(storage, args...), cell)
end

ParallelStencil.ParallelKernel.fill!_kernelabstractions(backend, A, x) = KernelAbstractions.fill!(backend, A, construct_cell(A, x))

check_datatype_kernelabstractions(args...) = check_datatype(args..., INT_KERNELABSTRACTIONS)