// engram_to_iq4nl.cpp - the Engram table a Strata pack can read, out of a llama.cpp Q4_K_M GGUF.
//
// `strata-dsv41` reads the Engram rows as MXFP8, Q6_K or IQ4_NL only (ds4_model.cu refuses anything else), and the
// ISTA-DAS 2-bit sets ship the table as a separate IQ4_NL GGUF that is not on this PC.  The Q4_K_M set on this PC does
// carry it (`blk.{1,14}.engram_embd.weight`, Q4_K, 256 values a row, 384M rows a layer), so this tool decodes those rows
// with ggml's own Q4_K decoder and re-encodes them with ggml's own IQ4_NL quantizer (ggml_quantize_chunk), and writes a
// GGUF with architecture "deepseek41-engram" that pack_gguf.py takes as --engram.
//
//   g++ -O2 -mavx2 -mfma -mf16c -o engram_to_iq4nl engram_to_iq4nl.cpp \
//       -I<llama.cpp>/ggml/include <build>/ggml/src/libggml-base.a <build>/ggml/src/libggml-cpu.a -lpthread -lm
//   ./engram_to_iq4nl --src <q4km dir> --out <dir>/engram-iq4nl.gguf
//
// Rows are streamed: peak RSS is a few MB whatever the table's size.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <algorithm>
#include <stdio.h>

#include "ggml.h"

namespace {

void pread_full(int fd, void* p, size_t n, uint64_t off) {
    uint8_t* d = (uint8_t*) p;
    while (n > 0) {
        const ssize_t r = pread(fd, d, n, (off_t) off);
        if (r <= 0) { std::fprintf(stderr, "pread %zu at %llu failed\n", n, (unsigned long long) off); std::exit(2); }
        d += r; n -= (size_t) r; off += (uint64_t) r;
    }
}
void pwrite_full(int fd, const void* p, size_t n, uint64_t off) {
    const uint8_t* d = (const uint8_t*) p;
    while (n > 0) {
        const ssize_t r = pwrite(fd, d, n, (off_t) off);
        if (r <= 0) { std::fprintf(stderr, "pwrite failed\n"); std::exit(2); }
        d += r; n -= (size_t) r; off += (uint64_t) r;
    }
}

struct TensorInfo {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dims;
    uint64_t off = 0;   // from the data section
    uint64_t bytes = 0;
};

struct GgufFile {
    std::string path;
    uint64_t data_start = 0;
    std::vector<TensorInfo> tensors;
    uint32_t align = 32;
};

uint64_t type_size(enum ggml_type t, const std::vector<uint64_t>& dims, uint64_t blck, size_t tsize) {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n / blck * tsize;
}

void rd(FILE* f, void* p, size_t n, const char* what) {
    if (std::fread(p, 1, n, f) != n) {
        std::fprintf(stderr, "read %s failed\n", what);
        std::exit(2);
    }
}

std::string rd_str(FILE* f) {
    uint64_t n = 0;
    rd(f, &n, 8, "str len");
    std::string s(n, '\0');
    if (n) rd(f, s.data(), n, "str");
    return s;
}

// the metadata block is skipped: this tool only needs the tensor table and the data offsets
void skip_val(FILE* f, uint32_t t) {
    switch (t) {
        case 0: case 1: { uint8_t v; rd(f, &v, 1, "kv"); break; }
        case 2: case 3: { uint16_t v; rd(f, &v, 2, "kv"); break; }
        case 4: case 5: case 6: { uint32_t v; rd(f, &v, 4, "kv"); break; }
        case 7: { uint8_t v; rd(f, &v, 1, "kv"); break; }
        case 10: case 11: case 12: { uint64_t v; rd(f, &v, 8, "kv"); break; }
        case 8: { (void) rd_str(f); break; }
        case 9: {
            uint32_t at; uint64_t n;
            rd(f, &at, 4, "arr"); rd(f, &n, 8, "arr");
            for (uint64_t i = 0; i < n; ++i) skip_val(f, at);
            break;
        }
        default: std::fprintf(stderr, "unknown metadata type %u\n", t); std::exit(2);
    }
}

GgufFile read_gguf(const std::string& path) {
    GgufFile g; g.path = path;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "%s: cannot open\n", path.c_str()); std::exit(2); }
    char magic[4]; rd(f, magic, 4, "magic");
    if (std::memcmp(magic, "GGUF", 4) != 0) { std::fprintf(stderr, "%s: not a GGUF\n", path.c_str()); std::exit(2); }
    uint32_t version; uint64_t nt, nkv;
    rd(f, &version, 4, "version"); rd(f, &nt, 8, "nt"); rd(f, &nkv, 8, "nkv");
    for (uint64_t i = 0; i < nkv; ++i) {
        const std::string k = rd_str(f);
        uint32_t t; rd(f, &t, 4, "kv type");
        if (k == "general.alignment" && t == 4) { uint32_t a; rd(f, &a, 4, "align"); g.align = a; continue; }
        skip_val(f, t);
    }
    for (uint64_t i = 0; i < nt; ++i) {
        TensorInfo ti;
        ti.name = rd_str(f);
        uint32_t nd; rd(f, &nd, 4, "nd");
        ti.dims.resize(nd);
        if (nd) rd(f, ti.dims.data(), 8 * nd, "dims");
        rd(f, &ti.type, 4, "type");
        rd(f, &ti.off, 8, "off");
        g.tensors.push_back(ti);
    }
    const long pos = std::ftell(f);
    g.data_start = (uint64_t) ((pos + g.align - 1) / g.align * g.align);
    std::fclose(f);
    return g;
}

void wr(FILE* f, const void* p, size_t n) {
    if (std::fwrite(p, 1, n, f) != n) { std::fprintf(stderr, "write failed\n"); std::exit(2); }
}
void wr_str(FILE* f, const std::string& s) {
    const uint64_t n = s.size();
    wr(f, &n, 8); wr(f, s.data(), n);
}

}  // namespace

int main(int argc, char** argv) {
    std::string src, out;
    int rows_per_chunk = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--src" && i + 1 < argc) src = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--chunk" && i + 1 < argc) rows_per_chunk = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s --src <q4km dir> --out <file.gguf>\n", argv[0]); return 2; }
    }
    if (src.empty() || out.empty()) { std::fprintf(stderr, "need --src and --out\n"); return 2; }

    // the two shards that hold one Engram table each (llama.cpp's split: names say which)
    struct Job { std::string shard; std::string tensor; };
    std::vector<Job> jobs;
    {
        FILE* p = popen(("ls " + src + "/*.gguf").c_str(), "r");
        if (!p) { std::fprintf(stderr, "cannot list %s\n", src.c_str()); return 2; }
        char line[4096];
        while (std::fgets(line, sizeof(line), p)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s.empty()) continue;
            GgufFile g = read_gguf(s);
            for (const auto& t : g.tensors)
                if (t.name.find("engram_embd.weight") != std::string::npos) jobs.push_back({s, t.name});
        }
        pclose(p);
    }
    if (jobs.empty()) { std::fprintf(stderr, "no engram_embd.weight in %s\n", src.c_str()); return 2; }
    std::fprintf(stderr, "engram tables: %zu\n", jobs.size());

    // the output's header: one metadata key, one tensor a table, IQ4_NL rows
    const uint64_t hd = 256;                       // values a row (the model's engram.head_dim)
    int out_fd = open(out.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { std::fprintf(stderr, "%s: cannot write\n", out.c_str()); return 2; }
    uint64_t pos = 0;
    auto w = [&](const void* p, size_t n) { pwrite_full(out_fd, p, n, pos); pos += n; };
    auto wstr = [&](const std::string& t) { const uint64_t n = t.size(); w(&n, 8); w(t.data(), n); };
    const uint32_t version = 3;
    uint64_t n_tensors = jobs.size(), n_kv = 1;
    w("GGUF", 4); w(&version, 4); w(&n_tensors, 8); w(&n_kv, 8);
    wstr("general.architecture"); { uint32_t t = 8; w(&t, 4); wstr("deepseek41-engram"); }
    std::vector<uint64_t> offs(jobs.size(), 0);
    {
        uint64_t off = 0;
        for (size_t j = 0; j < jobs.size(); ++j) {
            GgufFile g = read_gguf(jobs[j].shard);
            uint64_t rows = 0;
            for (const auto& t : g.tensors) if (t.name == jobs[j].tensor) rows = t.dims.empty() ? 0 : t.dims.back();
            if (rows == 0) { std::fprintf(stderr, "%s: %s has no rows\n", jobs[j].shard.c_str(), jobs[j].tensor.c_str()); return 2; }
            const uint64_t bytes = rows * (hd / 32 * 18);     // IQ4_NL: 32 values a block, 18 bytes
            off = (off + 31) / 32 * 32;
            offs[j] = off;
            off += bytes;
            wstr(jobs[j].tensor);
            uint32_t nd = 2; uint64_t d0 = hd, d1 = rows; uint32_t ty = 20;   // 20 = GGML_TYPE_IQ4_NL
            w(&nd, 4); w(&d0, 8); w(&d1, 8); w(&ty, 4); w(&offs[j], 8);
        }
        const uint64_t data_start = (pos + 31) / 32 * 32;
        if (data_start > pos) { std::vector<char> z((size_t) (data_start - pos), 0); w(z.data(), z.size()); }
    }
    const long out_data = (long) pos;

    // the rows: Q4_K -> float -> IQ4_NL.  The offsets on both sides are known, so N threads read their own ranges and
    // write their own slices (ggml's decoders are the cost; one thread managed 20k rows/s, too slow for 768M rows).
    const int threads = [] {
        const char* e = std::getenv("ENGRAM_THREADS");
        return e ? std::max(1, std::atoi(e)) : (int) std::thread::hardware_concurrency();
    }();
    const auto* tr_q4k = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (!tr_q4k || !tr_q4k->to_float) { std::fprintf(stderr, "ggml has no Q4_K decoder\n"); return 2; }
    for (size_t j = 0; j < jobs.size(); ++j) {
        GgufFile g = read_gguf(jobs[j].shard);
        const TensorInfo* ti = nullptr;
        for (const auto& t : g.tensors) if (t.name == jobs[j].tensor) ti = &t;
        if (!ti) { std::fprintf(stderr, "%s: %s vanished\n", jobs[j].shard.c_str(), jobs[j].tensor.c_str()); return 2; }
        if (ti->type != (uint32_t) GGML_TYPE_Q4_K) {
            std::fprintf(stderr, "%s: %s is type %u, not Q4_K\n", jobs[j].shard.c_str(), jobs[j].tensor.c_str(), ti->type);
            return 2;
        }
        const uint64_t rows = ti->dims.back();
        const int in_fd = open(jobs[j].shard.c_str(), O_RDONLY);
        if (in_fd < 0) { std::fprintf(stderr, "cannot read %s\n", jobs[j].shard.c_str()); return 2; }
        const uint64_t in_base = g.data_start + ti->off;
        const uint64_t row_in = hd / 256 * 144, row_out = hd / 32 * 18;
        const uint64_t out_base = out_data + (long) offs[j];
        std::fprintf(stderr, "%s: %llu rows, %d threads\n", ti->name.c_str(), (unsigned long long) rows, threads);
        std::atomic<uint64_t> done{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const uint64_t chunk = 4096;
                std::vector<uint8_t> rb((size_t) chunk * row_in);
                std::vector<float> fb((size_t) chunk * hd);
                std::vector<uint8_t> ib((size_t) chunk * row_out);
                for (uint64_t r = (uint64_t) t * chunk; r < rows; r += (uint64_t) threads * chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, rows - r);
                    pread_full(in_fd, rb.data(), n * row_in, in_base + r * row_in);
                    tr_q4k->to_float(rb.data(), fb.data(), (int64_t) n * (int64_t) hd);
                    const size_t wrote = ggml_quantize_chunk(GGML_TYPE_IQ4_NL, fb.data(), ib.data(), 0, (int64_t) n,
                                                             (int64_t) hd, nullptr);
                    pwrite_full(out_fd, ib.data(), wrote, out_base + r * row_out);
                    done += n;
                }
            });
        }
        for (auto& th : pool) th.join();
        close(in_fd);
        std::fprintf(stderr, "  done %llu rows\n", (unsigned long long) done.load());
    }
    close(out_fd);
    struct stat st{};
    stat(out.c_str(), &st);
    std::fprintf(stderr, "%s: %.1f GiB\n", out.c_str(), (double) st.st_size / (1 << 30));
    return 0;
}
