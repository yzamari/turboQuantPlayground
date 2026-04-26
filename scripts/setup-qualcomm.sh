#!/usr/bin/env bash
# setup-qualcomm.sh — one-shot setup for the TurboQuant Qualcomm stack.
#
# Detects what's installed, downloads / builds whatever's missing, and produces:
#   - cpp/build-android/                          (TurboQuant C++ library + bench CLI + tests)
#   - cpp/build-host/                             (host-side parity tests)
#   - external/llama.cpp/build-android/           (llama.cpp binaries for Android arm64)
#   - external/models/                            (Llama-3.2-1B + SmolVLM-256M GGUFs)
#   - android/app/build/outputs/apk/debug/        (the assistant APK)
#
# Then optionally pushes everything to a connected Android device.
#
# Usage:
#   scripts/setup-qualcomm.sh                # default: deps cpp llamacpp models app
#   scripts/setup-qualcomm.sh all            # also pushes to device + verifies on-device
#   scripts/setup-qualcomm.sh <step> [...]   # run only the listed steps in order
#
# Steps:
#   deps        — verify (and on macOS, brew-install) the toolchain
#   cpp         — build the TurboQuant C++ port (host + Android arm64)
#   llamacpp    — clone + build llama.cpp for Android arm64
#   models      — download Llama-3.2-1B + SmolVLM GGUFs (~1 GB total)
#   app         — build the Android assistant APK (Gradle 8.10.2 wrapper)
#   push        — adb-push all artifacts + models to a connected device
#   verify      — run tests on host + on device
#   all         — everything above, in that order
#
# Environment:
#   ANDROID_NDK_HOME    — path to NDK r26+. If unset, auto-detect under
#                         ~/Library/Android/sdk/ndk/ (macOS) or $ANDROID_HOME/ndk/.
#   JAVA_HOME           — JDK 17. If unset, auto-detect via /usr/libexec/java_home -v 17 (macOS).
#   GGUF_LLAMA_URL      — override the Llama-3.2-1B GGUF download URL.
#   GGUF_SMOLVLM_URL    — override the SmolVLM GGUF download URL.
#   GGUF_SMOLVLM_MMPROJ — override the SmolVLM mmproj download URL.
#   SKIP_BREW           — set to 1 to skip brew installation steps on macOS.

set -euo pipefail

# -----------------------------------------------------------------------------
# Repo + paths
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/external"
LLAMACPP_DIR="${EXTERNAL_DIR}/llama.cpp"
MODELS_DIR="${EXTERNAL_DIR}/models"
LLAMACPP_BUILD="${LLAMACPP_DIR}/build-android"
CPP_BUILD_ANDROID="${REPO_ROOT}/cpp/build-android"
CPP_BUILD_HOST="${REPO_ROOT}/cpp/build-host"
ANDROID_DIR="${REPO_ROOT}/android"
APK_PATH="${ANDROID_DIR}/app/build/outputs/apk/debug/app-debug.apk"

DEVICE_TMP="/data/local/tmp/llama"
DEVICE_APP_FILES="/sdcard/Android/data/com.yzamari.turboquant/files"

GGUF_LLAMA_URL="${GGUF_LLAMA_URL:-https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true}"
GGUF_LLAMA_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"

GGUF_SMOLVLM_URL="${GGUF_SMOLVLM_URL:-https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf?download=true}"
GGUF_SMOLVLM_MMPROJ="${GGUF_SMOLVLM_MMPROJ:-https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf?download=true}"
GGUF_SMOLVLM_FILE="SmolVLM-256M-Instruct-Q8_0.gguf"
GGUF_SMOLVLM_MMPROJ_FILE="mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'

step()  { echo "${BLUE}==> $*${RESET}"; }
ok()    { echo "${GREEN}    ✓ $*${RESET}"; }
warn()  { echo "${YELLOW}    ⚠ $*${RESET}"; }
die()   { echo "${RED}    ✗ $*${RESET}" >&2; exit 1; }

have()  { command -v "$1" >/dev/null 2>&1; }
is_macos()  { [[ "$(uname -s)" == "Darwin" ]]; }
is_linux()  { [[ "$(uname -s)" == "Linux"  ]]; }

# -----------------------------------------------------------------------------
# Step: deps — verify / install toolchain
# -----------------------------------------------------------------------------
step_deps() {
    step "Checking toolchain"

    # Required: cmake, ninja, git, curl, adb
    local missing=()
    for cmd in cmake ninja git curl adb; do
        if have "$cmd"; then
            ok "$cmd: $(command -v "$cmd")"
        else
            missing+=("$cmd")
        fi
    done

    # macOS auto-install via Homebrew
    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "missing: ${missing[*]}"
        if is_macos && [[ "${SKIP_BREW:-0}" != "1" ]] && have brew; then
            step "Installing missing tools via Homebrew"
            brew install "${missing[@]}" || warn "some packages may have failed"
        elif is_linux && have apt-get && [[ $EUID -eq 0 ]]; then
            apt-get update && apt-get install -y "${missing[@]}"
        else
            die "Please install: ${missing[*]}"
        fi
    fi

    # JDK 17 — required by Android Gradle Plugin
    if [[ -z "${JAVA_HOME:-}" ]]; then
        if is_macos && have /usr/libexec/java_home; then
            if JH=$(/usr/libexec/java_home -v 17 2>/dev/null); then
                export JAVA_HOME="$JH"
                ok "JAVA_HOME auto-detected: $JAVA_HOME"
            else
                if is_macos && [[ "${SKIP_BREW:-0}" != "1" ]] && have brew; then
                    step "Installing OpenJDK 17 via Homebrew"
                    brew install openjdk@17
                    export JAVA_HOME="/opt/homebrew/Cellar/openjdk@17/$(ls /opt/homebrew/Cellar/openjdk@17 | tail -1)/libexec/openjdk.jdk/Contents/Home"
                else
                    die "JDK 17 not found. Install OpenJDK 17 and set JAVA_HOME."
                fi
            fi
        else
            die "JAVA_HOME not set. Please install OpenJDK 17 and export JAVA_HOME."
        fi
    fi
    ok "JAVA_HOME=$JAVA_HOME"
    export PATH="$JAVA_HOME/bin:$PATH"

    # Android NDK
    if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
        for cand in \
            "${ANDROID_NDK:-}" \
            "${HOME}/Library/Android/sdk/ndk" \
            "${HOME}/Android/Sdk/ndk" \
            "${ANDROID_HOME:-}/ndk"
        do
            [[ -z "$cand" || ! -d "$cand" ]] && continue
            # Pick the highest-versioned NDK
            local newest
            newest=$(ls -1 "$cand" | sort | tail -1 || true)
            [[ -z "$newest" ]] && continue
            export ANDROID_NDK_HOME="$cand/$newest"
            ok "ANDROID_NDK_HOME auto-detected: $ANDROID_NDK_HOME"
            break
        done
        if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
            die "Android NDK not found. Install via Android Studio (SDK Manager → NDK r26+) or set ANDROID_NDK_HOME."
        fi
    else
        ok "ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
    fi

    # glslc for Vulkan shaders
    if have glslc; then
        ok "glslc: $(command -v glslc)"
    else
        warn "glslc not found — Vulkan backend will not be built. brew install glslang to enable."
    fi

    # Gradle (we'll use the wrapper but need a base gradle to bootstrap)
    if have gradle; then
        ok "gradle: $(gradle --version | head -1)"
    elif is_macos && [[ "${SKIP_BREW:-0}" != "1" ]] && have brew; then
        step "Installing Gradle via Homebrew (one-time, for wrapper bootstrap)"
        brew install gradle
    fi

    ok "Toolchain ready."
}

# -----------------------------------------------------------------------------
# Step: cpp — build TurboQuant C++ port
# -----------------------------------------------------------------------------
step_cpp() {
    step "Building TurboQuant C++ port"

    # Host build (parity tests)
    cmake -S "${REPO_ROOT}/cpp" -B "${CPP_BUILD_HOST}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTQ_BUILD_TESTS=ON
    cmake --build "${CPP_BUILD_HOST}" -j

    # Android arm64 build (NEON + OpenCL + Vulkan + bench)
    local android_args=(
        -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/cpp/cmake/toolchain-android-arm64.cmake"
        -DCMAKE_BUILD_TYPE=Release
        -DTQ_WITH_NEON=ON
        -DTQ_WITH_OPENCL=ON
    )
    if have glslc; then
        android_args+=(-DTQ_WITH_VULKAN=ON)
    fi
    cmake -S "${REPO_ROOT}/cpp" -B "${CPP_BUILD_ANDROID}" "${android_args[@]}"
    cmake --build "${CPP_BUILD_ANDROID}" -j

    ok "C++ port built. Artifacts:"
    ok "  host: ${CPP_BUILD_HOST}/bench/turboquant_bench"
    ok "  android: ${CPP_BUILD_ANDROID}/bench/turboquant_bench"
}

# -----------------------------------------------------------------------------
# Step: llamacpp — clone + build llama.cpp for Android arm64
# -----------------------------------------------------------------------------
step_llamacpp() {
    step "Building llama.cpp for Android arm64"

    if [[ ! -d "${LLAMACPP_DIR}/.git" ]]; then
        mkdir -p "${EXTERNAL_DIR}"
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMACPP_DIR}"
    else
        ok "llama.cpp checkout exists, skipping clone."
    fi

    cmake -S "${LLAMACPP_DIR}" -B "${LLAMACPP_BUILD}" \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-29 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENMP=OFF \
        -DGGML_CCACHE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=ON
    cmake --build "${LLAMACPP_BUILD}" -j

    # Also build our llama-turboquant-kv tool if the source is there
    if [[ -d "${LLAMACPP_DIR}/tools/turboquant_kv" ]]; then
        cmake --build "${LLAMACPP_BUILD}" --target llama-turboquant-kv -j || \
            warn "llama-turboquant-kv build failed (probably tools/CMakeLists.txt didn't include it)"
    fi

    ok "llama.cpp built. Key binaries:"
    ok "  ${LLAMACPP_BUILD}/bin/llama-completion"
    ok "  ${LLAMACPP_BUILD}/bin/llama-mtmd-cli"
    ok "  ${LLAMACPP_BUILD}/bin/llama-bench"

    # Copy .so files into the Android app's jniLibs so the APK build picks them up
    local jnilibs="${ANDROID_DIR}/app/src/main/jniLibs/arm64-v8a"
    mkdir -p "${jnilibs}"
    for so in libllama libggml libggml-base libggml-cpu libmtmd; do
        if [[ -f "${LLAMACPP_BUILD}/bin/${so}.so" ]]; then
            cp -f "${LLAMACPP_BUILD}/bin/${so}.so" "${jnilibs}/"
        fi
    done
    ok "Copied prebuilt .so files into ${jnilibs}/"
}

# -----------------------------------------------------------------------------
# Step: models — download GGUF model files
# -----------------------------------------------------------------------------
step_models() {
    step "Downloading models (~1 GB total)"
    mkdir -p "${MODELS_DIR}"

    download() {
        local url=$1 out=$2
        if [[ -f "${out}" && $(stat -f%z "${out}" 2>/dev/null || stat -c%s "${out}") -gt 1000000 ]]; then
            ok "$(basename "${out}") already present ($(($(stat -f%z "${out}" 2>/dev/null || stat -c%s "${out}") / 1048576)) MB)"
            return 0
        fi
        echo "    downloading $(basename "${out}") ..."
        curl -L --progress-bar -o "${out}" "${url}"
    }

    download "${GGUF_LLAMA_URL}"     "${MODELS_DIR}/${GGUF_LLAMA_FILE}"
    download "${GGUF_SMOLVLM_URL}"   "${MODELS_DIR}/${GGUF_SMOLVLM_FILE}"
    download "${GGUF_SMOLVLM_MMPROJ}" "${MODELS_DIR}/${GGUF_SMOLVLM_MMPROJ_FILE}"

    ok "Models in ${MODELS_DIR}/"
}

# -----------------------------------------------------------------------------
# Step: app — build the Android assistant APK
# -----------------------------------------------------------------------------
step_app() {
    step "Building Android assistant APK"
    cd "${ANDROID_DIR}"
    if [[ ! -f "./gradlew" ]]; then
        if have gradle; then
            gradle wrapper --gradle-version 8.10.2 --distribution-type all
        else
            die "gradle not installed; run \`scripts/setup-qualcomm.sh deps\` first."
        fi
    fi
    ./gradlew :app:assembleDebug
    cd "${REPO_ROOT}"
    ok "APK at ${APK_PATH}"
}

# -----------------------------------------------------------------------------
# Step: push — push everything to the connected Android device
# -----------------------------------------------------------------------------
step_push() {
    step "Pushing to connected device"
    if ! adb devices | grep -qE '\bdevice$'; then
        die "No authorized device connected. Plug in, tap 'Allow' on USB debugging, retry."
    fi
    adb shell "mkdir -p ${DEVICE_TMP}"

    # llama.cpp binaries + libs
    if [[ -d "${LLAMACPP_BUILD}/bin" ]]; then
        for f in llama-completion llama-simple-chat llama-mtmd-cli llama-bench llama-turboquant-kv \
                 libllama.so libggml.so libggml-base.so libggml-cpu.so libmtmd.so libllama-common.so; do
            if [[ -f "${LLAMACPP_BUILD}/bin/${f}" ]]; then
                adb push "${LLAMACPP_BUILD}/bin/${f}" "${DEVICE_TMP}/" >/dev/null
            fi
        done
        ok "llama.cpp binaries pushed to ${DEVICE_TMP}/"
    fi

    # TurboQuant bench
    if [[ -f "${CPP_BUILD_ANDROID}/bench/turboquant_bench" ]]; then
        adb push "${CPP_BUILD_ANDROID}/bench/turboquant_bench" "${DEVICE_TMP}/" >/dev/null
        ok "turboquant_bench pushed to ${DEVICE_TMP}/"
    fi

    # Tests + golden corpus
    if [[ -d "${REPO_ROOT}/cpp/tests/golden" ]]; then
        adb shell "mkdir -p ${DEVICE_TMP}/golden"
        adb push "${REPO_ROOT}/cpp/tests/golden/." "${DEVICE_TMP}/golden/" >/dev/null
        for t in tq_packing_test tq_smoke_test tq_parity_test; do
            [[ -f "${CPP_BUILD_ANDROID}/tests/${t}" ]] && \
                adb push "${CPP_BUILD_ANDROID}/tests/${t}" "${DEVICE_TMP}/" >/dev/null
        done
        ok "Tests + golden corpus pushed."
    fi

    # Models — push to both /data/local/tmp/llama and the app's external dir
    for model in "${GGUF_LLAMA_FILE}" "${GGUF_SMOLVLM_FILE}" "${GGUF_SMOLVLM_MMPROJ_FILE}"; do
        if [[ -f "${MODELS_DIR}/${model}" ]]; then
            adb push "${MODELS_DIR}/${model}" "${DEVICE_TMP}/" >/dev/null
        fi
    done
    ok "Models pushed to ${DEVICE_TMP}/"

    # APK install (if built)
    if [[ -f "${APK_PATH}" ]]; then
        adb install -r "${APK_PATH}" >/dev/null && ok "APK installed: com.yzamari.turboquant"
        adb shell "mkdir -p ${DEVICE_APP_FILES}" 2>/dev/null || true
        if [[ -f "${MODELS_DIR}/${GGUF_LLAMA_FILE}" ]]; then
            adb push "${MODELS_DIR}/${GGUF_LLAMA_FILE}" "${DEVICE_APP_FILES}/" >/dev/null && \
                ok "Llama-3.2-1B copied into app data dir."
        fi
    fi
}

# -----------------------------------------------------------------------------
# Step: verify — run tests + smoke checks
# -----------------------------------------------------------------------------
step_verify() {
    step "Verifying"

    # Host parity
    if [[ -d "${CPP_BUILD_HOST}" ]]; then
        ctest --test-dir "${CPP_BUILD_HOST}" --output-on-failure || warn "host ctest had failures"
    fi

    # Device parity
    if adb devices | grep -qE '\bdevice$'; then
        adb shell "${DEVICE_TMP}/tq_parity_test ${DEVICE_TMP}/golden" 2>&1 | tail -3 || true
        adb shell "cd ${DEVICE_TMP} && LD_LIBRARY_PATH=. ./llama-completion -m ${GGUF_LLAMA_FILE} -p 'Q: 2+2? A:' -n 8 -t 8 -c 256 --no-warmup" 2>&1 | tail -5 || warn "LLM smoke failed"
    fi

    ok "Verification done."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local steps=("$@")
    if [[ ${#steps[@]} -eq 0 ]]; then
        steps=(deps cpp llamacpp models app)
    elif [[ "${steps[0]}" == "all" ]]; then
        steps=(deps cpp llamacpp models app push verify)
    fi

    for s in "${steps[@]}"; do
        case "$s" in
            deps)     step_deps     ;;
            cpp)      step_cpp      ;;
            llamacpp) step_llamacpp ;;
            models)   step_models   ;;
            app)      step_app      ;;
            push)     step_push     ;;
            verify)   step_verify   ;;
            *) die "unknown step: $s (use one of: deps cpp llamacpp models app push verify all)" ;;
        esac
    done

    echo
    step "Done."
    echo "  • C++ port:       ${CPP_BUILD_ANDROID}/"
    echo "  • llama.cpp:      ${LLAMACPP_BUILD}/bin/"
    echo "  • models:         ${MODELS_DIR}/"
    [[ -f "${APK_PATH}" ]] && echo "  • APK:            ${APK_PATH}"
    echo
    echo "Next:"
    echo "  scripts/setup-qualcomm.sh push verify   # push to device + run on-device tests"
}

main "$@"
