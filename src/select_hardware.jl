##
@doc replace(ParallelKernel.SELECT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function select_hardware(hardware::Symbol)
    ParallelKernel.select_hardware(hardware)
end

##
@doc replace(ParallelKernel.CURRENT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function current_hardware()
    ParallelKernel.current_hardware()
end
