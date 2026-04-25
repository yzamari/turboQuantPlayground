# Build-time helper that wraps embed_resource() so we can call it from
# add_custom_command(... -P embed_runner.cmake). The parent CMakeLists passes
# EMBED_INPUT, EMBED_OUTPUT, EMBED_SYMBOL via -D.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/embed_resource.cmake)
embed_resource("${EMBED_INPUT}" "${EMBED_OUTPUT}" "${EMBED_SYMBOL}")
