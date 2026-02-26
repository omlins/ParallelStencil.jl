##
@doc replace(ParallelKernel.SELECT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function select_hardware(caller::Module, hardware::Symbol; package::Symbol=ParallelKernel.get_package(caller))
    ParallelKernel.select_hardware(caller, hardware; package=package)
end

##
@doc replace(ParallelKernel.CURRENT_HARDWARE_DOC, "ParallelKernel" => "ParallelStencil"
) function current_hardware(caller::Module; package::Symbol=ParallelKernel.get_package(caller))
    ParallelKernel.current_hardware(caller; package=package)
end

##
@doc replace(ParallelKernel.SELECT_HARDWARE_MACRO_DOC, "ParallelKernel" => "ParallelStencil")
macro select_hardware(hardware)
    return esc(:(ParallelStencil.ParallelKernel.@select_hardware $hardware))
end

##
@doc replace(ParallelKernel.CURRENT_HARDWARE_MACRO_DOC, "ParallelKernel" => "ParallelStencil")
macro current_hardware()
    return esc(:(ParallelStencil.ParallelKernel.@current_hardware))
end
