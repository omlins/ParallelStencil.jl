@parallel_indices (ix) function Diffusion.memcopy!(B::Data.Array, A::Data.Array)
    B[ix] = A[ix]
    return
end
