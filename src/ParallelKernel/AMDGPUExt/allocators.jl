## RUNTIME ALLOCATOR FUNCTIONS

ParallelStencil.ParallelKernel.zeros_amdgpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype_amdgpu(T); AMDGPU.zeros(T, args...))  # (blocklength is ignored if neither celldims nor celltype is set)
ParallelStencil.ParallelKernel.ones_amdgpu(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype_amdgpu(T); AMDGPU.ones(T, args...))   # ...
ParallelStencil.ParallelKernel.rand_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = ROCArray(rand_cpu(T, blocklength, args...))           # ...
ParallelStencil.ParallelKernel.falses_amdgpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = AMDGPU.falses(args...)                                # ...
ParallelStencil.ParallelKernel.trues_amdgpu(::Type{T}, blocklength, args...) where {T<:Bool}                        = AMDGPU.trues(args...)                                 # ...
ParallelStencil.ParallelKernel.fill_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = ROCArray(fill_cpu(T, blocklength, args...))           # ...

ParallelStencil.ParallelKernel.zeros_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_amdgpu(T); fill_amdgpu(T, blocklength, 0, args...))
ParallelStencil.ParallelKernel.ones_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_amdgpu(T); fill_amdgpu(T, blocklength, 1, args...))
ParallelStencil.ParallelKernel.rand_amdgpu(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype_amdgpu(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B, AMDGPU.ROCArray{eltype(T),3}}(AMDGPU.ROCArray(Base.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/(blocklen)))), dims))   # TODO: use AMDGPU.rand! instead of AMDGPU.rand once it supports Enums: rand_amdgpu(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype_amdgpu(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B, AMDGPU.ROCArray{eltype(T),3}}(AMDGPU.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/(blocklen))), dims))
ParallelStencil.ParallelKernel.rand_amdgpu(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_amdgpu(T, blocklength, dims)
ParallelStencil.ParallelKernel.falses_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_amdgpu(T, blocklength, false, args...)
ParallelStencil.ParallelKernel.trues_amdgpu(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_amdgpu(T, blocklength, true, args...)

function ParallelStencil.ParallelKernel.fill_amdgpu(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
    if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
    check_datatype_amdgpu(T, Bool, Enum)
    if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
    elseif (length(x) == length(T)) cell = convert(T, x)
    else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
    end
    return CellArrays.fill!(ROCCellArray{T,B}(undef, args...), cell)
end

ParallelStencil.ParallelKernel.fill_amdgpu!(A, x) = AMDGPU.fill!(A, construct_cell(A, x))

check_datatype_amdgpu(args...) = check_datatype(args..., INT_AMDGPU)