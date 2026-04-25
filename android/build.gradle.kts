// Root build script — TurboQuant Android demo.
// Plugin versions are declared here and applied by the :app module.

plugins {
    id("com.android.application")             version "8.5.2"  apply false
    id("org.jetbrains.kotlin.android")        version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
