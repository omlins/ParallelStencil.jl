@parallel_indices (I...) ndims=(2,3) function memcopy!(B::Data.Array{T,$ndims}, A::Data.Array{T,$ndims}) where {T}
    B[I...] = A[I...]
    return
end


# NOTE: this is equivalent to the following:

# @parallel_indices (I...) ndims=2 function memcopy!(B::Data.Array{T,$ndims}, A::Data.Array{T,$ndims}) where {T}
#     B[I...] = A[I...]
#     return
# end

# @parallel_indices (I...) ndims=3 function memcopy!(B::Data.Array{T,$ndims}, A::Data.Array{T,$ndims}) where {T}
#     B[I...] = A[I...]
#     return
# end


# NOTE: and equivalent to the following:

# @parallel_indices (I...) ndims=2 function memcopy!(B::Data.Array{T,2}, A::Data.Array{T,2}) where {T}
#     B[I...] = A[I...]
#     return
# end

# @parallel_indices (I...) ndims=3 function memcopy!(B::Data.Array{T,3}, A::Data.Array{T,3}) where {T}
#     B[I...] = A[I...]
#     return
# end
