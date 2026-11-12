#pragma once

// N parts run as N threads of one process, for the distributed multigrid's tests (#259 D2): each
// collective call posts this part's data, waits for the others, reads theirs, and waits again
// before returning, so that no buffer changes while another part reads it. A part that fails
// drops out of the barrier and raises `failed`; the others stop at their next call.

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "autan/partition/partition.hpp"
#include "autan/runtime/exact_sum.hpp"
#include "autan/solvers/distributed_multigrid.hpp"

namespace autan::test {

struct Hub {
  explicit Hub(std::int32_t n)
      : barrier(n), posted(static_cast<std::size_t>(n)), values(static_cast<std::size_t>(n)) {}
  struct Post {
    const HaloPattern* pattern = nullptr;
    std::vector<double*> components;
  };
  std::barrier<> barrier;
  std::vector<Post> posted;
  std::vector<std::vector<double>> values;  // sums, maxima and gathers
  std::atomic<bool> failed{false};
};

class ThreadRanks final : public solvers::LevelCollective {
 public:
  ThreadRanks(Hub& hub, std::int32_t rank) : hub_(hub), rank_(rank) {}

  void forward(const HaloPattern& pattern, std::size_t n_owned,
               std::span<double* const> components) override {
    // LevelExchange's contract, HaloExchange's limit.
    if (components.size() > 16) throw std::length_error("more than 16 components in one call");
    hub_.posted[at(rank_)] = {&pattern, {components.begin(), components.end()}};
    wait();
    for (std::size_t i = 0; i < pattern.neighbours.size(); ++i) {
      const std::int32_t q = pattern.neighbours[i];
      const Hub::Post& theirs = hub_.posted[at(q)];
      const HaloPattern& hq = *theirs.pattern;
      const auto j = at(std::ranges::find(hq.neighbours, rank_) - hq.neighbours.begin());
      auto send = at(hq.send_offsets[j]);
      for (auto k = at(pattern.recv_offsets[i]); k < at(pattern.recv_offsets[i + 1]); ++k, ++send) {
        for (std::size_t c = 0; c < components.size(); ++c)
          components[c][n_owned + k] = theirs.components[c][at(hq.send_cells[send])];
      }
    }
    hub_.barrier.arrive_and_wait();
  }

  // The exact sum of every part's terms, rounded once, as the runtime's Ranks::sum.
  double sum(std::span<const double> terms) override {
    runtime::ExactSum s;
    for (const auto& v : gather(terms)) s.add(v);
    return s.value();
  }

  double max(double value) override {
    const double mine = std::isnan(value) ? std::numeric_limits<double>::infinity() : value;
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& v : gather(std::span<const double>(&mine, 1))) m = std::max(m, v.front());
    return m;
  }

  std::vector<double> all_gather(std::span<const double> values) override {
    std::vector<double> all;
    for (const auto& v : gather(values)) all.insert(all.end(), v.begin(), v.end());
    return all;
  }

 private:
  static std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

  void wait() {
    hub_.barrier.arrive_and_wait();
    if (hub_.failed) throw std::runtime_error("another part failed");
  }

  std::vector<std::vector<double>> gather(std::span<const double> mine) {
    hub_.values[at(rank_)].assign(mine.begin(), mine.end());
    wait();
    std::vector<std::vector<double>> all = hub_.values;
    hub_.barrier.arrive_and_wait();
    return all;
  }

  Hub& hub_;
  std::int32_t rank_;
};

}  // namespace autan::test
