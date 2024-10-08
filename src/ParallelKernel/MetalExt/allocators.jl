## RUNTIME ALLOCATOR FUNCTIONS

ParallelStencil.ParallelKernel.zeros_metal(::Type{T}, blocklength, args...) where {T<:Number}                     = (check_datatype_metal(T); Metal.zeros(T, args...))  # (blocklength is ignored if neither celldims nor celltype is set)
ParallelStencil.ParallelKernel.ones_metal(::Type{T}, blocklength, args...) where {T<:Number}                      = (check_datatype_metal(T); Metal.ones(T, args...))
ParallelStencil.ParallelKernel.rand_metal(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = (check_datatype_metal(T); MtlArray(rand_cpu(T, blocklength, args...)))
ParallelStencil.ParallelKernel.falses_metal(::Type{T}, blocklength, args...) where {T<:Bool}                      = Metal.falses(args...)
ParallelStencil.ParallelKernel.trues_metal(::Type{T}, blocklength, args...) where {T<:Bool}                       = Metal.trues(args...)
ParallelStencil.ParallelKernel.fill_metal(::Type{T}, blocklength, args...) where {T<:Union{Number,Enum}}          = MtlArray(fill_cpu(T, blocklength, args...))

ParallelStencil.ParallelKernel.zeros_metal(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_metal(T); fill_metal(T, blocklength, 0, args...))
ParallelStencil.ParallelKernel.ones_metal(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = (check_datatype_metal(T); fill_metal(T, blocklength, 1, args...))
ParallelStencil.ParallelKernel.rand_metal(::Type{T}, ::Val{B},    dims)    where {T<:Union{SArray,FieldArray}, B} = (check_datatype_metal(T, Bool, Enum); blocklen = (B == 0) ? prod(dims) : B; CellArray{T,length(dims),B, Metal.MtlArray{eltype(T),3}}(Metal.MtlArray(Base.rand(eltype(T), blocklen, prod(size(T)), ceil(Int,prod(dims)/(blocklen))), dims)))
ParallelStencil.ParallelKernel.rand_metal(::Type{T}, blocklength, dims...) where {T<:Union{SArray,FieldArray}}    = rand_metal(T, blocklength, dims)
ParallelStencil.ParallelKernel.falses_metal(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_metal(T, blocklength, false, args...)
ParallelStencil.ParallelKernel.trues_metal(::Type{T}, blocklength, args...) where {T<:Union{SArray,FieldArray}}    = fill_metal(T, blocklength, true, args...)

# function ParallelStencil.ParallelKernel.fill_metal(::Type{T}, ::Val{B}, x, args...) where {T <: Union{SArray,FieldArray}, B}
#     if (!(eltype(x) <: Number) || (eltype(x) == Bool)) && (eltype(x) != eltype(T)) @ArgumentError("fill: the (element) type of argument 'x' is not a normal number type ($(eltype(x))), but does not match the obtained (default) 'eltype' ($(eltype(T))); automatic conversion to $(eltype(T)) is therefore not attempted. Set the keyword argument 'eltype' accordingly to the element type of 'x' or pass an 'x' of a different (element) type.") end
#     check_datatype_metal(T, Bool, Enum)
#     if     (length(x) == 1)         cell = convert(T, fill(convert(eltype(T), x), size(T)))
#     elseif (length(x) == length(T)) cell = convert(T, x)
#     else                            @ArgumentError("fill: argument 'x' contains the wrong number of elements ($(length(x))). It must be a scalar or contain the number of elements defined by 'celldims'.")
#     end
#     return CellArrays.fill!(MtlCellArray{T,B}(undef, args...), cell)
# end

# ParallelStencil.ParallelKernel.fill_metal!(A, x) = Metal.fill!(A, construct_cell(A, x))

check_datatype_metal(args...) = check_datatype(args..., INT_METAL)