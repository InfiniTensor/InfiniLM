if(NOT DEFINED HOST_DIR OR NOT IS_DIRECTORY "${HOST_DIR}")
    message(FATAL_ERROR "`HOST_DIR` must point to an existing host object directory")
endif()

file(GLOB_RECURSE _host_objects "${HOST_DIR}/objects-*/*.o")

foreach(_obj IN LISTS _host_objects)
    get_filename_component(_obj_name "${_obj}" NAME)
    set(_dst "${HOST_DIR}/${_obj_name}")
    if(EXISTS "${_dst}")
        file(REMOVE "${_dst}")
    endif()
    file(RENAME "${_obj}" "${_dst}")
endforeach()
