@doc ParallelKernel.OVERLAP_DOC
macro overlap(args...)
    check_initialized(__module__);
    esc(:(ParallelStencil.ParallelKernel.@overlap($(args...))));
end
