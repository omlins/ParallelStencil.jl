@doc replace(ParallelKernel.HIDE_COMMUNICATION_DOC, "@init_parallel_kernel" => "@init_parallel_stencil")
macro hide_communication(args...)
    check_initialized(__module__);
    esc(:(ParallelStencil.ParallelKernel.@hide_communication($(args...))));
end
@doc replace(ParallelKernel.GET_PRIORITY_STREAM_DOC, "@init_parallel_kernel" => "@init_parallel_stencil")
macro get_priority_stream(args...)
    check_initialized(__module__);
    esc(:(ParallelStencil.ParallelKernel.@get_priority_stream($(args...))));
end
@doc replace(ParallelKernel.GET_STREAM_DOC, "@init_parallel_kernel" => "@init_parallel_stencil")
macro get_stream(args...)
    check_initialized(__module__);
    esc(:(ParallelStencil.ParallelKernel.@get_stream($(args...))));
end
