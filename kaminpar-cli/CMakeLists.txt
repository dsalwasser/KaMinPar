set(KAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    kaminpar_arguments.cc
    kaminpar_arguments.h)
set(DKAMINPAR_CLI_SOURCE_FILES
    CLI11.h
    dkaminpar_arguments.cc
    dkaminpar_arguments.h)

add_library(kaminpar_cli ${KAMINPAR_CLI_SOURCE_FILES})
add_library(KaMinPar::KaMinParCLI ALIAS kaminpar_cli)

target_link_libraries(kaminpar_cli PUBLIC kaminpar_shm)
target_compile_options(kaminpar_cli PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (TARGET kaminpar_dist)
    add_library(kaminpar_dist_cli ${DKAMINPAR_CLI_SOURCE_FILES})
    add_library(KaMinPar::dKaMinParCLI ALIAS kaminpar_dist_cli)

    target_link_libraries(kaminpar_dist_cli PUBLIC kaminpar_dist)
endif ()
