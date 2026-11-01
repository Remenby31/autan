#include "autan/physics/manufactured.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace autan::physics {

namespace {

enum class Op : std::uint8_t {
  constant,
  slot,  // x, y, z or a subexpression
  neg,
  add,
  sub,
  mul,
  div,
  pow,
  sqrt,
  sin,
  cos,
  exp,
  log,
};

struct Node {
  Op op;
  int a = -1;
  int b = -1;
  double value = 0.0;
  int slot = -1;
};

// Recursive-descent reader of one expression into `nodes`; `slots` maps names to slot indices.
class Parser {
 public:
  Parser(std::string text, const std::map<std::string, int>& slots, std::vector<Node>& nodes)
      : s_(std::move(text)), slots_(slots), nodes_(nodes) {}

  int parse() {
    const int root = sum();
    skip();
    if (i_ != s_.size()) fail("unexpected character");
    return root;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::invalid_argument(std::string("manufactured solution expression: ") + what +
                                " at position " + std::to_string(i_) + " of '" + s_ +
                                "'");
  }
  void skip() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_])) != 0) ++i_;
  }
  bool eat(char c) {
    skip();
    if (i_ < s_.size() && s_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }
  int add(Node n) {
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size()) - 1;
  }
  int sum() {
    int left = product();
    for (;;) {
      if (eat('+')) {
        left = add({.op = Op::add, .a = left, .b = product()});
      } else if (eat('-')) {
        left = add({.op = Op::sub, .a = left, .b = product()});
      } else {
        return left;
      }
    }
  }
  int product() {
    int left = unary();
    for (;;) {
      if (eat('*')) {
        left = add({.op = Op::mul, .a = left, .b = unary()});
      } else if (eat('/')) {
        left = add({.op = Op::div, .a = left, .b = unary()});
      } else {
        return left;
      }
    }
  }
  int unary() {
    if (eat('-')) return add({.op = Op::neg, .a = unary()});
    return primary();
  }
  int primary() {
    skip();
    if (eat('(')) {
      const int e = sum();
      if (!eat(')')) fail("expected ')'");
      return e;
    }
    if (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) != 0 || s_[i_] == '.')) {
      std::size_t j = i_;
      while (j < s_.size() &&
             (std::isdigit(static_cast<unsigned char>(s_[j])) != 0 || s_[j] == '.' || s_[j] == 'e' ||
              s_[j] == 'E' ||
              ((s_[j] == '+' || s_[j] == '-') && j > i_ && (s_[j - 1] == 'e' || s_[j - 1] == 'E'))))
        ++j;
      const std::string literal(s_.substr(i_, j - i_));
      char* end = nullptr;
      const double v = std::strtod(literal.c_str(), &end);
      if (end != literal.c_str() + literal.size()) fail("invalid number");
      i_ = j;
      return add({.op = Op::constant, .value = v});
    }
    std::size_t j = i_;
    while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) != 0 || s_[j] == '_'))
      ++j;
    if (j == i_) fail("expected a number, a name or '('");
    const std::string name(s_.substr(i_, j - i_));
    i_ = j;
    static const std::map<std::string, Op> functions = {
        {"pow", Op::pow}, {"sqrt", Op::sqrt}, {"sin", Op::sin},
        {"cos", Op::cos}, {"exp", Op::exp},   {"log", Op::log}};
    if (const auto f = functions.find(name); f != functions.end()) {
      if (!eat('(')) fail("expected '(' after a function");
      Node n{.op = f->second, .a = sum()};
      if (f->second == Op::pow) {
        if (!eat(',')) fail("pow takes two arguments");
        n.b = sum();
      }
      if (!eat(')')) fail("expected ')'");
      return add(n);
    }
    if (name == "pi") return add({.op = Op::constant, .value = std::numbers::pi});
    const auto slot = slots_.find(name);
    if (slot == slots_.end()) fail("unknown name");
    return add({.op = Op::slot, .slot = slot->second});
  }

  std::string s_;
  std::size_t i_ = 0;
  const std::map<std::string, int>& slots_;
  std::vector<Node>& nodes_;
};

double eval(const std::vector<Node>& nodes, int k, const std::vector<double>& slots) {
  const Node& n = nodes[static_cast<std::size_t>(k)];
  auto a = [&] { return eval(nodes, n.a, slots); };
  auto b = [&] { return eval(nodes, n.b, slots); };
  switch (n.op) {
    case Op::constant:
      return n.value;
    case Op::slot:
      return slots[static_cast<std::size_t>(n.slot)];
    case Op::neg:
      return -a();
    case Op::add:
      return a() + b();
    case Op::sub:
      return a() - b();
    case Op::mul:
      return a() * b();
    case Op::div:
      return a() / b();
    case Op::pow:
      return std::pow(a(), b());
    case Op::sqrt:
      return std::sqrt(a());
    case Op::sin:
      return std::sin(a());
    case Op::cos:
      return std::cos(a());
    case Op::exp:
      return std::exp(a());
    case Op::log:
      return std::log(a());
  }
  return 0.0;
}

}  // namespace

struct Manufactured::Group {
  std::vector<Node> nodes;
  std::vector<int> subexpressions;  // root of slot 3 + k
  std::map<std::string, int> outputs;
};

Manufactured::Manufactured(const std::string& json_text) {
  const auto doc = nlohmann::json::parse(json_text);
  if (doc.value("schema", "") != "autan.mms/1") {
    throw std::invalid_argument("not an autan.mms/1 document");
  }
  physics_ = doc.at("model").at("physics").get<std::string>();
  const auto& turbulence = doc.at("model").at("turbulence");
  turbulence_ = turbulence.is_null() ? "" : turbulence.get<std::string>();
  for (const auto& [k, v] : doc.at("parameters").items()) parameters_[k] = v.get<double>();
  for (const char* name : {"fields", "gradients", "conserved", "sources", "auxiliary"}) {
    if (!doc.contains(name)) continue;
    const auto& g = doc.at(name);
    auto group = std::make_unique<Group>();
    std::map<std::string, int> slots = {{"x", 0}, {"y", 1}, {"z", 2}};
    for (const auto& pair : g.at("subexpressions")) {
      Parser parser(pair.at(1).get<std::string>(), slots, group->nodes);
      group->subexpressions.push_back(parser.parse());
      slots[pair.at(0).get<std::string>()] = static_cast<int>(slots.size());
    }
    for (const auto& [k, v] : g.at("outputs").items()) {
      Parser parser(v.get<std::string>(), slots, group->nodes);
      group->outputs[k] = parser.parse();
    }
    groups_[name] = std::move(group);
  }
}

Manufactured::~Manufactured() = default;
Manufactured::Manufactured(Manufactured&&) noexcept = default;
Manufactured& Manufactured::operator=(Manufactured&&) noexcept = default;

std::vector<std::vector<double>> Manufactured::evaluate(
    const std::string& group, std::span<const std::string> names,
    std::span<const std::array<double, 3>> points) const {
  const auto g = groups_.find(group);
  if (g == groups_.end()) {
    throw std::invalid_argument("manufactured solution has no group '" + group + "'");
  }
  const Group& grp = *g->second;
  std::vector<int> roots;
  for (const auto& name : names) {
    const auto o = grp.outputs.find(name);
    if (o == grp.outputs.end()) {
      throw std::invalid_argument("manufactured solution group '" + group + "' has no '" + name +
                                  "'");
    }
    roots.push_back(o->second);
  }
  std::vector<std::vector<double>> out(names.size(), std::vector<double>(points.size()));
  std::vector<double> slots(3 + grp.subexpressions.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    slots[0] = points[i][0];
    slots[1] = points[i][1];
    slots[2] = points[i][2];
    for (std::size_t k = 0; k < grp.subexpressions.size(); ++k) {
      slots[3 + k] = eval(grp.nodes, grp.subexpressions[k], slots);
    }
    for (std::size_t k = 0; k < roots.size(); ++k) out[k][i] = eval(grp.nodes, roots[k], slots);
  }
  return out;
}

}  // namespace autan::physics
