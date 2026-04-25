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
    ndkVersion = "26.3.11579264"

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
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DANDROID_ARM_NEON=ON",
                    "-DTQ_BUILD_TESTS=OFF",
                    "-DTQ_BUILD_BENCH=OFF",
                    "-DTQ_WITH_CPU_SCALAR=ON",
                    "-DTQ_WITH_NEON=ON",
                    "-DTQ_WITH_OPENCL=ON",
                    "-DTQ_WITH_VULKAN=ON",
                    "-DTQ_WITH_QNN=OFF",
                )
                cppFlags += listOf("-std=c++17", "-O3", "-fno-exceptions", "-fno-rtti")
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
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")

    val composeBom = platform("androidx.compose:compose-bom:2024.09.03")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
