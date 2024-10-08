function(add_shm_benchmark target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        KaMinPar::KaMinPar 
        KaMinPar::KaMinParCLI 
        KaMinPar::KaMinParIO 
        KaGen::KaGen)
    target_compile_options(${target} PRIVATE ${KAMINPAR_WARNING_FLAGS})
    message(STATUS "Enabled benchmark: ${target}")
endfunction()

# Shared-memory benchmarks
add_shm_benchmark(shm_input_benchmark shm_input_benchmark.cc)
add_shm_benchmark(shm_label_propagation_benchmark shm_label_propagation_benchmark.cc)
add_shm_benchmark(shm_refinement_benchmark shm_refinement_benchmark.cc)
#add_shm_benchmark(shm_gain_cache_benchmark shm_gain_cache_benchmark.cc)
add_shm_benchmark(shm_compressed_graph_benchmark shm_compressed_graph_benchmark.cc)
add_shm_benchmark(shm_variable_length_codec_benchmark shm_variable_length_codec_benchmark.cc)

if (KAMINPAR_BUILD_DISTRIBUTED)
    function(add_dist_benchmark target)
        add_shm_benchmark(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE 
            KaMinPar::dKaMinPar 
            KaMinPar::dKaMinParCLI
            KaMinPar::dKaMinParIO)
    endfunction()

    # Distributed benchmarks
    add_dist_benchmark(dist_coloring_benchmark dist_coloring_benchmark.cc)
    add_dist_benchmark(dist_contraction_benchmark dist_contraction_benchmark.cc)
    add_dist_benchmark(dist_refinement_benchmark dist_refinement_benchmark.cc)
    add_dist_benchmark(dist_coarsening_benchmark dist_coarsening_benchmark.cc)
    add_dist_benchmark(dist_block_clustering_benchmark dist_block_clustering_benchmark.cc)
endif ()
