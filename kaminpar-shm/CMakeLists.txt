file(GLOB_RECURSE KAMINPAR_SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(kaminpar_shm ${KAMINPAR_SHM_SOURCE_FILES})
target_include_directories(kaminpar_shm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar_shm PUBLIC kaminpar_common)
target_compile_options(kaminpar_shm PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (Sparsehash_FOUND)
    target_link_libraries(kaminpar_shm PUBLIC Sparsehash::Sparsehash)
endif ()

# If we can find Mt-KaHyPar, make it available as an option for refinement
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(kaminpar_shm PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(kaminpar_shm PUBLIC mtkahypar)
endif ()
