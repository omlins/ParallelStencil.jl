# NOTE: these three definitions can be condensed in one in future.

@parallel_indices (I...) ndims=3 function memcopy!(B::Data.Array{T,3}, A::Data.Array{T,3}) where {T}
    B[I...] = A[I...]
    return
end

@parallel_indices (I...) ndims=2 function memcopy!(B::Data.Array{T,2}, A::Data.Array{T,2}) where {T}
    B[I...] = A[I...]
    return
end

@parallel_indices (I...) ndims=1 function memcopy!(B::Data.Array{T,1}, A::Data.Array{T,1}) where {T}
    B[I...] = A[I...]
    return
end