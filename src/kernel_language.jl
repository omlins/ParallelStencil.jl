@doc replace(ParallelKernel.GRIDDIM_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro gridDim(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.gridDim(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.BLOCKIDX_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro blockIdx(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.blockIdx(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.BLOCKDIM_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro blockDim(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.blockDim(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.THREADIDX_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro threadIdx(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.threadIdx(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SYNCTHREADS_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro sync_threads(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.sync_threads(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SHAREDMEM_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro sharedMem(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_sharedMem(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.sharedMem(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.FORALL_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro ∀(args...)
	check_initialized(__module__)
	ParallelKernel.checkforallargs(args...)
	esc(:(ParallelStencil.ParallelKernel.@∀($(args...))))
end

@doc replace(replace(ParallelKernel.PKSHOW_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"), "pk_show" => "ps_show") macro ps_show(args...)
	check_initialized(__module__)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.pk_show(__module__, args...; package=pkg))
end

@doc replace(replace(ParallelKernel.PKPRINTLN_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"), "pk_println" => "ps_println") macro ps_println(args...)
	check_initialized(__module__)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.pk_println(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.WARPSIZE_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro warpsize(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.warpsize(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.LANEID_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro laneid(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.laneid(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.ACTIVE_MASK_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro active_mask(args...)
	check_initialized(__module__)
	ParallelKernel.checknoargs(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.active_mask(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SHFL_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro shfl_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_shfl_sync(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.shfl_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SHFL_UP_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro shfl_up_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_shfl_up_down_xor(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.shfl_up_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SHFL_DOWN_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro shfl_down_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_shfl_up_down_xor(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.shfl_down_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.SHFL_XOR_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro shfl_xor_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_shfl_up_down_xor(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.shfl_xor_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.VOTE_ANY_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro vote_any_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_vote(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.vote_any_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.VOTE_ALL_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro vote_all_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_vote(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.vote_all_sync(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.VOTE_BALLOT_SYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil") macro vote_ballot_sync(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_vote(args...)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel.vote_ballot_sync(__module__, args...; package=pkg))
end
