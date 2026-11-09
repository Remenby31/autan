# Header-only libraries every build needs, vendored so that a build without the tests downloads
# nothing (the tests still download Catch2)
# (README.md: where each file comes from, and its SHA-256 in SHA256SUMS).
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(
  nlohmann_json
  SYSTEM
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json/include
)

add_library(autan_picosha2 INTERFACE)
add_library(autan::picosha2 ALIAS autan_picosha2)
target_include_directories(autan_picosha2 SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/picosha2)
