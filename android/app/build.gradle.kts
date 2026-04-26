// :app — TurboQuant on-device benchmark UI.
//
// - Builds the C++ core via externalNativeBuild (CMake).
// - Compose / Material 3 UI; minSdk 29 (Android 10) so we have a wide ARMv8 base.
// - arm64-v8a only — TurboQuant targets 64-bit Snapdragon NEON / Adreno.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace  = "com.yzamari.turboquant"
    compileSdk = 34
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.yzamari.turboquant"
        minSdk        = 29
        targetSdk     = 34
        versionCode   = 1
        versionName   = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                // Qualcomm QNN/HTP backend is opt-in: build only if the dev
                // exports QNN_SDK_ROOT (env or gradle.properties), otherwise
                // libturboquant builds without it and create_best_backend()
                // skips QnnHtp at runtime. Either source works; Gradle
                // property has priority so it can be set per-machine in
                // local.properties without polluting the environment.
                val qnnSdkRoot: String? =
                    (project.findProperty("turboquant.qnnSdkRoot") as String?)
                        ?: System.getenv("QNN_SDK_ROOT")
                val qnnEnabled = !qnnSdkRoot.isNullOrBlank()

                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DANDROID_ARM_NEON=ON",
                    "-DTQ_BUILD_TESTS=OFF",
                    "-DTQ_BUILD_BENCH=OFF",
                    "-DTQ_WITH_CPU_SCALAR=ON",
                    "-DTQ_WITH_NEON=ON",
                    "-DTQ_WITH_OPENCL=OFF",
                    "-DTQ_WITH_VULKAN=OFF",
                    "-DTQ_WITH_QNN=" + (if (qnnEnabled) "ON" else "OFF"),
                )
                if (qnnEnabled) {
                    arguments += "-DQNN_SDK_ROOT=$qnnSdkRoot"
                    println(
                        "[turboquant] QNN/HTP backend ENABLED " +
                            "(QNN_SDK_ROOT=$qnnSdkRoot). Make sure " +
                            "lib/aarch64-android/*.so + " +
                            "lib/hexagon-v75/unsigned/*Skel.so " +
                            "are present under that root."
                    )
                }
                // Note: exceptions/rtti must be enabled because the cpp/ core
                // throws std::invalid_argument from kv_cache.cpp & quantizer.cpp.
                cppFlags += listOf("-std=c++17", "-O3")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path    = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
        // Extract jniLibs at install time so Runtime.exec() can invoke our
        // bundled binaries (we ship llama-mtmd-cli as libllama-mtmd-cli.so).
        jniLibs {
            useLegacyPackaging = true
            // libOpenCL.so in jniLibs/ is the Khronos ICD-loader stub used at
            // *link time* only — at runtime we want the device's vendor loader
            // (e.g. /vendor/lib64/libOpenCL.so on Snapdragon) which actually
            // knows how to find libOpenCL_adreno.so. Bundling our stub wins
            // over the vendor lib in the linker namespace and silently CPU-
            // falls back. Exclude it from the APK and rely on
            // <uses-native-library android:name="libOpenCL.so"/> in
            // AndroidManifest to grant the namespace access. The library still
            // exists on disk in jniLibs for CMake's link step.
            excludes += "**/libOpenCL.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.6")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // CameraX — backs the Live VLM camera feed (LiveCameraScreen). Pinned at
    // 1.4.x because that's the first stable line with ImageProxy.toBitmap()
    // built-in (no manual YUV→RGB), which keeps the per-frame conversion off
    // our hot path.
    val cameraxVersion = "1.4.0"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    val composeBom = platform("androidx.compose:compose-bom:2024.09.03")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
