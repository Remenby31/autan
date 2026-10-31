#pragma once

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>

#include "autan/core/error.hpp"

namespace autan::io {

static_assert(std::endian::native == std::endian::little,
              "binary mesh formats are read as little-endian");

/// Reads a whole file into memory; throws E0130 when it cannot.
std::string read_file(const std::string& path);

/// Forward-only reader over a file held in memory, shared by the text and
/// binary parts of the mesh formats. Errors carry the path and line.
class Cursor {
 public:
  Cursor(std::string_view data, std::string path) : data_(data), path_(std::move(path)) {}

  [[nodiscard]] bool at_end() const { return pos_ >= data_.size(); }
  [[nodiscard]] std::size_t position() const { return pos_; }
  [[nodiscard]] std::size_t size() const { return data_.size(); }
  [[nodiscard]] std::size_t remaining() const {
    return data_.size() - std::min(pos_, data_.size());
  }

  /// Fails unless `count` items of at least `bytes_each` bytes fit in the
  /// rest of the file, so that hostile sizes never reach an allocation.
  void check_fits(std::uint64_t count, std::uint64_t bytes_each) const {
    if (bytes_each != 0 && count > remaining() / bytes_each) {
      fail("declared size " + std::to_string(count) + " exceeds the rest of the file");
    }
  }
  [[nodiscard]] const std::string& path() const { return path_; }

  void skip_space() {
    while (pos_ < data_.size() && is_space(data_[pos_])) ++pos_;
  }

  [[nodiscard]] char peek() {
    skip_space();
    return at_end() ? '\0' : data_[pos_];
  }

  /// Next character without skipping whitespace.
  [[nodiscard]] char peek_raw() const { return at_end() ? '\0' : data_[pos_]; }

  void advance(std::size_t n = 1) {
    if (pos_ + n > data_.size()) fail("unexpected end of file");
    pos_ += n;
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool accept(char c) {
    if (peek() != c) return false;
    ++pos_;
    return true;
  }

  /// Consumes `word` (after whitespace) if it is next.
  bool accept(std::string_view word) {
    skip_space();
    if (data_.substr(pos_, word.size()) != word) return false;
    pos_ += word.size();
    return true;
  }

  void expect(std::string_view word) {
    if (!accept(word)) fail("expected '" + std::string(word) + "'");
  }

  /// Run of characters up to whitespace or one of `stops`.
  std::string_view token(std::string_view stops = "") {
    skip_space();
    const std::size_t begin = pos_;
    while (pos_ < data_.size() && !is_space(data_[pos_]) &&
           stops.find(data_[pos_]) == std::string_view::npos) {
      ++pos_;
    }
    if (pos_ == begin) fail("expected a token");
    return data_.substr(begin, pos_ - begin);
  }

  /// Rest of the current line, without the line break.
  std::string_view line() {
    const std::size_t begin = pos_;
    while (pos_ < data_.size() && data_[pos_] != '\n') ++pos_;
    std::string_view out = data_.substr(begin, pos_ - begin);
    if (pos_ < data_.size()) ++pos_;
    if (!out.empty() && out.back() == '\r') out.remove_suffix(1);
    return out;
  }

  template <typename T>
  T integer(int base = 10, std::string_view stops = "()") {
    const std::string_view t = token(stops);
    T value{};
    const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value, base);
    if (ec != std::errc() || end != t.data() + t.size()) {
      fail("expected an integer, found '" + std::string(t) + "'");
    }
    return value;
  }

  double real(std::string_view stops = "()") {
    const std::string_view t = token(stops);
    double value{};
    const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value);
    if (ec != std::errc() || end != t.data() + t.size()) {
      fail("expected a number, found '" + std::string(t) + "'");
    }
    return value;
  }

  /// Copies `count` little-endian values of type T from the raw stream.
  template <typename T>
  void binary(T* out, std::size_t count) {
    const std::size_t bytes = count * sizeof(T);
    if (count != 0 && bytes / count != sizeof(T)) fail("binary block too large");
    if (data_.size() - pos_ < bytes) fail("binary block runs past the end of the file");
    std::memcpy(out, data_.data() + pos_, bytes);
    pos_ += bytes;
  }

  template <typename T>
  T binary() {
    T value{};
    binary(&value, 1);
    return value;
  }

  [[noreturn]] void fail(const std::string& message,
                         ErrorCode code = ErrorCode::io_malformed) const {
    const std::size_t end = std::min(pos_, data_.size());
    const auto line_number =
        1 + std::count(data_.begin(), data_.begin() + static_cast<std::ptrdiff_t>(end), '\n');
    throw Error(code, path_ + ":" + std::to_string(line_number) + ": " + message);
  }

  [[noreturn]] void unsupported(const std::string& message) const {
    fail(message, ErrorCode::io_unsupported);
  }

 private:
  static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  }

  std::string_view data_;
  std::string path_;
  std::size_t pos_ = 0;
};

}  // namespace autan::io
