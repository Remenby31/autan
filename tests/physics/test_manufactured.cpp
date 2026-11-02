// Manufactured solutions read from autan.mms/1 documents (autan-vv's export of `euler`).
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <pthread.h>

#include "autan/physics/manufactured.hpp"

namespace {

std::string read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

const std::filesystem::path euler_mms{AUTAN_EULER_MMS};

// Runs `body` on a thread of `bytes` of stack, rethrowing what it throws.
void on_stack(std::size_t bytes, const std::function<void()>& body) {
  struct Call {
    const std::function<void()>* body;
    std::exception_ptr error;
  } call{&body, nullptr};
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, bytes);
  pthread_t thread;
  const int rc = pthread_create(
      &thread, &attr,
      [](void* arg) -> void* {
        auto* c = static_cast<Call*>(arg);
        try {
          (*c->body)();
        } catch (...) {
          c->error = std::current_exception();
        }
        return nullptr;
      },
      &call);
  pthread_attr_destroy(&attr);
  REQUIRE(rc == 0);
  pthread_join(thread, nullptr);
  if (call.error) std::rethrow_exception(call.error);
}

}  // namespace

TEST_CASE("the euler solution evaluates to the fields of Roy et al. (2004)", "[manufactured]") {
  const autan::physics::Manufactured mms(read(euler_mms));
  CHECK(mms.physics() == "compressible");
  CHECK(mms.turbulence() == "inviscid");
  const std::vector<std::array<double, 3>> points = {{0.3, 0.6, 0.9}, {0.0, 1.0, 0.25}};
  const std::array<std::string, 2> names = {"rho", "p"};
  const auto w = mms.evaluate("fields", names, points);
  const double pi = std::numbers::pi;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto [x, y, z] = points[i];
    // vv/src/autan_vv/mms/solutions.py, ROY_SUBSONIC_3D.
    const double rho = 1.0 + 0.15 * std::sin(pi * x) - 0.1 * std::cos(0.5 * pi * y) +
                       0.08 * std::sin(0.75 * pi * z);
    const double p = 1e5 - 3e4 * std::cos(pi * x) + 2e4 * std::sin(1.25 * pi * y) +
                     1e4 * std::cos(0.75 * pi * z);
    CHECK(std::abs(w[0][i] - rho) <= 16 * 2.2e-16 * rho);
    CHECK(std::abs(w[1][i] - p) <= 16 * 2.2e-16 * p);
  }
}

TEST_CASE("the euler mass source is the divergence of the momentum", "[manufactured]") {
  const autan::physics::Manufactured mms(read(euler_mms));
  const std::array<double, 3> x0 = {0.37, 0.52, 0.81};
  const double h = 1e-4;
  std::vector<std::array<double, 3>> points;
  for (std::size_t a = 0; a < 3; ++a) {
    for (const double s : {-h, h}) {
      auto x = x0;
      x[a] += s;
      points.push_back(x);
    }
  }
  const std::array<std::string, 3> momentum = {"rhou", "rhov", "rhow"};
  const auto m = mms.evaluate("conserved", momentum, points);
  double divergence = 0.0;
  for (std::size_t a = 0; a < 3; ++a) divergence += (m[a][2 * a + 1] - m[a][2 * a]) / (2 * h);
  const std::array<std::string, 1> mass = {"rho"};
  const std::vector<std::array<double, 3>> at = {x0};
  const double source = mms.evaluate("sources", mass, at)[0][0];
  // Central differences: O(h^2) truncation, relative to the momentum gradient's scale.
  CHECK(std::abs(source - divergence) <= 1e-6 * 100.0 * std::numbers::pi * 20.0);
}

TEST_CASE("an expression outside the autan.mms/1 grammar is refused", "[manufactured]") {
  auto doc = [](const std::string& expr) {
    return R"({"schema": "autan.mms/1", "model": {"physics": "compressible", "turbulence": null},
               "parameters": {}, "fields": {"subexpressions": [], "outputs": {"rho": ")" +
           expr + R"("}}})";
  };
  CHECK_NOTHROW(autan::physics::Manufactured(doc("pow(x, 2) + exp(-y) / (1 + pi)")));
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("system(1)")), std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("x ** 2")), std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("t0 + 1")), std::invalid_argument);
  auto with_subexpressions = [](const std::string& subexpressions, const std::string& expr) {
    return R"({"schema": "autan.mms/1", "model": {"physics": "compressible", "turbulence": null},
               "parameters": {}, "fields": {"subexpressions": )" +
           subexpressions + R"(, "outputs": {"rho": ")" + expr + R"("}}})";
  };
  const std::array<std::string, 1> rho = {"rho"};
  const std::vector<std::array<double, 3>> origin = {{0.0, 0.0, 0.0}};
  const autan::physics::Manufactured chained(
      with_subexpressions(R"([["a", "1"], ["b", "a + 2"], ["c", "b * 3"]])", "c - a"));
  CHECK(chained.evaluate("fields", rho, origin)[0][0] == 8.0);
  CHECK_THROWS_AS(
      autan::physics::Manufactured(with_subexpressions(R"([["a", "1"], ["a", "2"]])", "a")),
      std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(with_subexpressions(R"([["a", "1", "2"]])", "a")),
                  std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(with_subexpressions(R"({"a": "1"})", "a")),
                  std::invalid_argument);
}

TEST_CASE("expressions as deep as the contract allows read and evaluate on a small stack",
          "[manufactured]") {
  // case-schema.md, `verification`: the runtime refuses an expression whose tree is deeper than
  // 1024. The runtime runs on the process's main thread (8 MiB of stack on Linux and macOS); these
  // read and evaluate on 1 MiB. A left-deep chain of 2^17 terms, beyond the contract, evaluates
  // too: the evaluation does not recurse.
  auto repeat = [](const std::string& s, std::size_t n) {
    std::string out;
    for (std::size_t i = 0; i < n; ++i) out += s;
    return out;
  };
  auto doc = [](const std::string& expr) {
    return R"({"schema": "autan.mms/1", "model": {"physics": "compressible", "turbulence": null},
               "parameters": {}, "fields": {"subexpressions": [], "outputs": {"rho": ")" +
           expr + R"("}}})";
  };
  const std::array<std::string, 1> rho = {"rho"};
  const std::vector<std::array<double, 3>> at = {{0.5, 0.0, 0.0}};
  const std::array<std::pair<std::string, double>, 4> cases = {{
      {"x" + repeat(" + x", 1023), 512.0},                           // depth 1024, left-deep
      {repeat("(", 1023) + "x" + repeat(")", 1023), 0.5},            // depth 1024, parentheses
      {repeat("-", 1023) + "x", -0.5},                               // depth 1024, unary minus
      {"x" + repeat(" + x", (std::size_t{1} << 17U) - 1), 65536.0},  // depth 2^17
  }};
  for (const auto& [expr, expected] : cases) {
    double value = 0.0;
    on_stack(std::size_t{1} << 20U, [&] {
      const autan::physics::Manufactured mms(doc(expr));
      value = mms.evaluate("fields", rho, at)[0][0];
    });
    CHECK(value == expected);
  }
}
