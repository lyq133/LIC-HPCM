#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "rans64.h"

namespace py = pybind11;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class ubransEncoder {
public:
  ubransEncoder() = default;

  ubransEncoder(const ubransEncoder &) = delete;
  ubransEncoder(ubransEncoder &&) = delete;
  ubransEncoder &operator=(const ubransEncoder &) = delete;
  ubransEncoder &operator=(ubransEncoder &&) = delete;

  void encode_with_indexes(const py::array_t<int32_t> &symbols,
                           const py::array_t<int32_t> &indexes,
                           const py::array_t<int32_t> &cdfs,
                           const py::array_t<int32_t> &cdfs_sizes,
                           const py::array_t<int32_t> &offsets);

  py::bytes flush();

private:
  std::vector<RansSymbol> _syms;
};

class ubransDecoder {
public:
  ubransDecoder() = default;

  ubransDecoder(const ubransDecoder &) = delete;
  ubransDecoder(ubransDecoder &&) = delete;
  ubransDecoder &operator=(const ubransDecoder &) = delete;
  ubransDecoder &operator=(ubransDecoder &&) = delete;

  void set_stream(const std::string &encoded);

  py::array_t<int32_t>
  decode_stream(const py::array_t<int32_t> &indexes,
                const py::array_t<int32_t> &cdfs,
                const py::array_t<int32_t> &cdfs_sizes,
                const py::array_t<int32_t> &offsets);

private:
  Rans64State _rans;
  std::string _stream;
  uint32_t *_ptr;
};
