#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {

/* We only run this in debug mode as its costly... */
// void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
//                  const std::vector<int> &cdfs_sizes) {
//   for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
//     assert(cdfs[i][0] == 0);
//     assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
//     for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
//       assert(cdfs[i][j + 1] > cdfs[i][j]);
//     }
//   }
// }

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  // assert(nbits <= 16);
  // assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    // Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    // Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} // namespace

void ubransEncoder::encode_with_indexes(
    const py::array_t<int32_t> &symbols, const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  // assert(cdfs.shape()[0] == cdfs_sizes.shape()[0]);

  // backward loop on symbols from the end;
  for (unsigned int i = 0; i < symbols.shape()[0]; ++i) {
    const int32_t cdf_idx = indexes.at(i);
    // assert(cdf_idx >= 0);
    // assert(cdf_idx < cdfs.size());

    // const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes.at(cdf_idx) - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    int32_t value = symbols.at(i) - offsets.at(cdf_idx);

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    // assert(value >= 0);
    // assert(value < cdfs_sizes[cdf_idx] - 1);

    _syms.push_back({static_cast<uint16_t>(cdfs.at(cdf_idx, value)),
                     static_cast<uint16_t>(cdfs.at(cdf_idx, value+1) - cdfs.at(cdf_idx, value)),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

py::bytes ubransEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  // assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

void ubransDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
  // assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

py::array_t<int32_t> ubransDecoder::decode_stream(
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets) {
  // assert(cdfs.shape()[0] == cdfs_sizes.shape()[0]);

  py::array_t<int32_t> output(indexes.shape()[0]);

  // assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.shape()[0]); ++i) {
    const int32_t cdf_idx = indexes.at(i);
    // assert(cdf_idx >= 0);
    // assert(cdf_idx < cdfs.size());

    // const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes.at(cdf_idx) - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets.at(cdf_idx);

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_begin = cdfs.data() + cdfs.index_at(cdf_idx, 0);
    const auto cdf_end = cdf_begin + cdfs_sizes.at(cdf_idx);
    const auto it = std::find_if(
        cdf_begin, cdf_end, [cum_freq](uint32_t v) { return v > cum_freq; });
    // assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf_begin, it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdfs.at(cdf_idx, s),
                     cdfs.at(cdf_idx, s + 1) - cdfs.at(cdf_idx, s), precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        // assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }
    output.mutable_data()[i] = value + offset;
  }
  return output;
}

PYBIND11_MODULE(unbounded_ans, m) {
  m.attr("__name__") = "unbounded_ans";

  m.doc() = "unbounded range Asymmetric Numeral System python bindings";

  py::class_<ubransEncoder>(m, "ubransEncoder")
      .def(py::init<>())
      .def("flush", &ubransEncoder::flush)
      .def("encode_with_indexes", &ubransEncoder::encode_with_indexes);

  py::class_<ubransDecoder>(m, "ubransDecoder")
      .def(py::init<>())
      .def("set_stream", &ubransDecoder::set_stream)
      .def("decode_stream", &ubransDecoder::decode_stream);
}
