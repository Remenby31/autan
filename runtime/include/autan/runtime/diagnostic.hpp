#pragma once

#include <stdexcept>
#include <string>

namespace autan::runtime {

/// A diagnostic in the shape of docs/contracts/case-schema.md#diagnostics.
/// Codes come from the SDK registry (autan.errors): E0203, E0704, E0705,
/// E0706, D0107, and the mesh core's E01xx.
struct Diagnostic {
  std::string code;
  std::string message;
  std::string path;       ///< document path for input errors, empty otherwise
  std::string data_json;  ///< JSON object text of `data`, "{}" when empty

  /// The diagnostic as one JSON object; severity, fix and doc come from the
  /// registry docs/contracts/diagnostics.json.
  [[nodiscard]] std::string json() const;
};

/// Raised before a run starts; autan-run reports it on stderr and exits 2.
class InputError : public std::runtime_error {
 public:
  explicit InputError(Diagnostic diagnostic)
      : std::runtime_error(diagnostic.code + ": " + diagnostic.message),
        diagnostic_(std::move(diagnostic)) {}
  [[nodiscard]] const Diagnostic& diagnostic() const noexcept { return diagnostic_; }

 private:
  Diagnostic diagnostic_;
};

}  // namespace autan::runtime
