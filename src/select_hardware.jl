##
@doc replace(ParallelKernel.SELECT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function select_hardware(hardware::Symbol)
    return ParallelKernel.select_hardware(hardware)
end

##
@doc replace(ParallelKernel.CURRENT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function current_hardware()
    return ParallelKernel.current_hardware()
end
