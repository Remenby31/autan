#pragma once

#include <array>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace autan::physics {

/// A manufactured solution read from its autan.mms/1 document (docs/validation/mms.md): each
/// group (`fields`, `conserved`, `sources`, ...) is a list of named expressions in x, y, z,
/// evaluated in FP64. Expressions follow the document's grammar only: + - * /, unary minus,
/// pow, sqrt, sin, cos, exp, log, decimal literals, pi and earlier subexpressions; anything else
/// throws std::invalid_argument when the document is read.
class Manufactured {
 public:
  explicit Manufactured(const std::string& json_text);
  ~Manufactured();
  Manufactured(Manufactured&&) noexcept;
  Manufactured& operator=(Manufactured&&) noexcept;

  [[nodiscard]] const std::string& physics() const noexcept { return physics_; }
  [[nodiscard]] const std::string& turbulence() const noexcept { return turbulence_; }
  [[nodiscard]] const std::map<std::string, double>& parameters() const noexcept {
    return parameters_;
  }

  /// Values of `names` of `group` at each point (x, y, z): out[k][i] for names[k] at point i.
  [[nodiscard]] std::vector<std::vector<double>> evaluate(
      const std::string& group, std::span<const std::string> names,
      std::span<const std::array<double, 3>> points) const;

 private:
  struct Group;
  std::string physics_, turbulence_;
  std::map<std::string, double> parameters_;
  std::map<std::string, std::unique_ptr<Group>> groups_;
};

}  // namespace autan::physics
