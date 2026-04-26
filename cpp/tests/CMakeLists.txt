# Lightweight test harness — uses a 50-line hand-rolled assertion library
# (cpp/tests/tq_test.hpp) so the core has no GoogleTest / Catch2 dependency.
# This keeps the automotive-toolchain story simple.

add_executable(tq_packing_test packing_test.cpp)
target_link_libraries(tq_packing_test PRIVATE turboquant)
target_include_directories(tq_packing_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME packing COMMAND tq_packing_test)

add_executable(tq_smoke_test smoke_test.cpp)
target_link_libraries(tq_smoke_test PRIVATE turboquant)
add_test(NAME smoke COMMAND tq_smoke_test)

# Byte-exact parity test against the Python reference. The .bin files are
# produced by `cpp/tools/gen_golden.py`; we hand the absolute path through a
# compile definition so the binary works under `ctest` without needing argv.
add_executable(tq_parity_test parity_test.cpp)
target_link_libraries(tq_parity_test PRIVATE turboquant)
target_compile_definitions(tq_parity_test PRIVATE
    TQ_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME parity COMMAND tq_parity_test)

# attention_turboquant() — Path 2.1c Step 1 stateless attention bridge.
# Parity vs hand-rolled FP32 reference at BH=8, D=128, n_kv=512, key_bits=3.
add_executable(tq_attention_turboquant_test attention_turboquant_test.cpp)
target_link_libraries(tq_attention_turboquant_test PRIVATE turboquant)
add_test(NAME attention_turboquant COMMAND tq_attention_turboquant_test)

# QnnSigLipBlock parity — Path C, Phase 1 scaffold.
#
# Compares one transformer block of SigLIP-base on QNN HTP against the
# FP32 reference produced by cpp/tools/siglip_block_ref.py. The .bin
# golden has to be regenerated on a Linux box with QAIRT — see the
# "SigLIP scaffold (Phase 1)" section of cpp/backends/qnn_htp/README.md
# and the .github/workflows/siglip-convert.yml workflow.
#
# Intentionally NOT added to ctest at Phase 1 — QnnSigLipBlock::forward()
# is currently a std::logic_error stub, so `ctest` would always go red.
# Run manually:
#     ./cpp/build-host/cpp/tests/tq_qnn_siglip_block_test
# Once steps 1-7 in qnn_siglip.cpp are filled in, uncomment the
# add_test() line below and the test joins the regular suite.
add_executable(tq_qnn_siglip_block_test qnn_siglip_block_test.cpp)
target_link_libraries(tq_qnn_siglip_block_test PRIVATE turboquant)
target_compile_definitions(tq_qnn_siglip_block_test PRIVATE
    TQ_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
# add_test(NAME qnn_siglip_block COMMAND tq_qnn_siglip_block_test)  # Phase 1: stub throws
