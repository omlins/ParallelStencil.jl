@doc replace(ParallelKernel.ZEROS_DOC,              "@init_parallel_kernel" => "@init_parallel_stencil") macro zeros(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	eltype, celldims, celltype, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@zeros")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._zeros(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.ONES_DOC,               "@init_parallel_kernel" => "@init_parallel_stencil") macro ones(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	eltype, celldims, celltype, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@ones")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._ones(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.RAND_DOC,               "@init_parallel_kernel" => "@init_parallel_stencil") macro rand(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	eltype, celldims, celltype, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@rand")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._rand(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.FALSES_DOC,             "@init_parallel_kernel" => "@init_parallel_stencil") macro falses(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	celldims, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@falses")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._falses(__module__, posargs...; celldims=celldims, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.TRUES_DOC,              "@init_parallel_kernel" => "@init_parallel_stencil") macro trues(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	celldims, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:celldims, :blocklength), "@trues")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._trues(__module__, posargs...; celldims=celldims, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.FILL_DOC,               "@init_parallel_kernel" => "@init_parallel_stencil") macro fill(args...)
	check_initialized(__module__)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	eltype, celldims, celltype, blocklength = ParallelKernel.extract_kwargvalues(kwargs_expr, (:eltype, :celldims, :celltype, :blocklength), "@fill")
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._fill(__module__, posargs...; eltype=eltype, celldims=celldims, celltype=celltype, blocklength=blocklength, package=pkg))
end

@doc replace(ParallelKernel.FILL!_DOC,              "@init_parallel_kernel" => "@init_parallel_stencil") macro fill!(args...)
	check_initialized(__module__)
	pkg = get_package(__module__)
	esc(ParallelStencil.ParallelKernel._fill!(__module__, args...; package=pkg))
end

@doc replace(ParallelKernel.CELLTYPE_DOC,           "@init_parallel_kernel" => "@init_parallel_stencil") macro CellType(args...)
	check_initialized(__module__)
	ParallelKernel.checkargs_CellType(args...)
	posargs, kwargs_expr = ParallelKernel.split_args(args)
	eltype, fieldnames, dims, parametric = ParallelKernel.extract_kwargvalues(kwargs_expr, (:eltype, :fieldnames, :dims, :parametric), "@CellType")
	esc(ParallelStencil.ParallelKernel._CellType(__module__, posargs...; eltype=eltype, fieldnames=fieldnames, dims=dims, parametric=parametric))
end
