# Convert any binary file into a C source file declaring:
#   const unsigned char <symbol>[N] = { ... };
#   const size_t        <symbol>_size = N;
#
# Used to embed codebook JSONs and GPU kernel source into the library so the
# runtime has no filesystem dependency (required for automotive deployment).
function(embed_resource INPUT_FILE OUTPUT_FILE SYMBOL_NAME)
    if(NOT EXISTS "${INPUT_FILE}")
        message(FATAL_ERROR "embed_resource: input file does not exist: ${INPUT_FILE}")
    endif()

    file(READ "${INPUT_FILE}" CONTENT HEX)
    string(LENGTH "${CONTENT}" HEX_LEN)
    math(EXPR BYTES "${HEX_LEN} / 2")

    set(BODY "")
    set(I 0)
    set(LINE "")
    while(I LESS HEX_LEN)
        string(SUBSTRING "${CONTENT}" ${I} 2 BYTE)
        set(LINE "${LINE}0x${BYTE},")
        math(EXPR I "${I} + 2")
        math(EXPR COL "(${I} / 2) % 16")
        if(COL EQUAL 0)
            set(BODY "${BODY}${LINE}\n    ")
            set(LINE "")
        endif()
    endwhile()
    if(NOT LINE STREQUAL "")
        set(BODY "${BODY}${LINE}\n")
    endif()

    set(SRC "/* Auto-generated from ${INPUT_FILE} — do not edit. */\n")
    set(SRC "${SRC}#include <stddef.h>\n\n")
    set(SRC "${SRC}const unsigned char ${SYMBOL_NAME}[] = {\n    ${BODY}};\n")
    set(SRC "${SRC}const size_t ${SYMBOL_NAME}_size = ${BYTES};\n")
    file(WRITE "${OUTPUT_FILE}" "${SRC}")
endfunction()
