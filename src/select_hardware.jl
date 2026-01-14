##
@doc replace(ParallelKernel.SELECT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function select_hardware(caller::Module, hardware::Symbol)
    ParallelKernel.select_hardware(caller, hardware)
end

##
@doc replace(ParallelKernel.CURRENT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function current_hardware(caller::Module)
    ParallelKernel.current_hardware(caller)
end
