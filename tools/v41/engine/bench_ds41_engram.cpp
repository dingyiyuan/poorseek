// tools/v41/engine/bench_ds41_engram.cpp - the Engram row-read bench (CPU + disk, no GPU).
//
// Two jobs:
//   1. raw disk probes on the real shard - random 4 KiB page reads at high queue depth, via pread / O_DIRECT / mmap:
//      this is the ceiling the row reader can reach on this exFAT volume.
//   2. the module itself - old (rows_reference) vs new (rows / start+wait) rows for n tokens of real hashes from a
//      .i32 token file, timed at n = 1, 512, 1024, 4096, with a --check mode that diffs the two byte for byte.
//
// usage:
//   bench_ds41_engram --gguf v41-all-h.gguf --dir <safetensors-dir> --seq seq.i32
//                     [--tokens 1,512,1024,4096] [--threads 16] [--reps 3] [--check]
//   bench_ds41_engram --raw FILE [--raw-n 20000] [--raw-threads 16] [--raw-mode pread|odirect|mmap] [--raw-lo 0.0] [--raw-hi 1.0]
#include "ds41_engram.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace strata;
using namespace strata::v41;

static double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static std::vector<int32_t> load_i32(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "bench: cannot open %s\n", p.c_str()); std::exit(2); }
    std::vector<int32_t> v;
    int32_t x;
    while (f.read(reinterpret_cast<char *>(&x), 4)) v.push_back(x);
    return v;
}

// ---------------------------------------------------------------- raw disk probes

static uint64_t xorshift(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

struct RawHit { uint64_t n = 0, bytes = 0; };

// n random `size`-byte reads spread over [lo, hi), `threads` workers, each walking a private offset stream.
// `drop` evicts [lo,hi) from the page cache first (fadvise DONTNEED on a window - cheap, and a cold read then).
static RawHit raw_bench(const std::string & file, int64_t n, int threads, const std::string & mode, uint64_t lo, uint64_t hi,
                        int64_t size, bool drop) {
    struct stat st;
    if (::stat(file.c_str(), &st) != 0) { std::fprintf(stderr, "bench: stat %s failed\n", file.c_str()); std::exit(2); }
    const uint64_t fsize = (uint64_t) st.st_size;
    if (hi == 0 || hi > fsize) hi = fsize;
    const int flags = O_RDONLY | (mode == "odirect" ? O_DIRECT : 0);
    const int fd = ::open(file.c_str(), flags);
    if (fd < 0) { std::fprintf(stderr, "bench: open %s (%s) failed\n", file.c_str(), mode.c_str()); std::exit(2); }
    if (drop && mode != "odirect") {
        const uint64_t win = 4ull << 30;   // windowed: one giant DONTNEED over the file is pathologically slow afterwards
        for (uint64_t o = lo; o < hi; o += win)
            ::posix_fadvise(fd, (off_t) (o & ~(uint64_t) 4095), (off_t) std::min<uint64_t>(win, hi - o), POSIX_FADV_DONTNEED);
    }
    void * map = nullptr;
    if (mode == "mmap") {
        map = ::mmap(nullptr, (size_t) fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) { std::fprintf(stderr, "bench: mmap failed\n"); std::exit(2); }
    }
    const uint64_t span = (hi > lo + (uint64_t) size) ? (hi - lo - (uint64_t) size) : 0;
    std::atomic<uint64_t> hits { 0 }, bad { 0 };
    auto worker = [&](int tid) {
        uint64_t rng = 0x9e3779b97f4a7c15ull ^ (uint64_t) (tid + 1) * 0x100000001b3ull;
        uint8_t * buf = nullptr;
        if (mode != "mmap") { if (::posix_memalign((void **) &buf, 4096, (size_t) size)) std::exit(2); }
        uint64_t mine = 0;
        for (int64_t i = tid; i < n; i += threads) {
            const uint64_t off = (lo + xorshift(rng) % (span ? span : 1)) & ~(uint64_t) 4095;
            if (mode == "mmap") {
                volatile uint8_t v = ((const uint8_t *) map)[off];
                (void) v;
                mine++;
            } else {
                const ssize_t got = ::pread(fd, buf, (size_t) size, (off_t) off);
                if (got == size) mine++; else bad.fetch_add(1);
            }
        }
        hits.fetch_add(mine);
    };
    std::vector<std::thread> ts;
    const double t0 = now_s();
    for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
    for (auto & t : ts) t.join();
    const double dt = now_s() - t0;
    RawHit h;
    h.n = hits.load();
    h.bytes = h.n * (uint64_t) size;
    std::printf("raw %-8s size=%5" PRId64 " n=%" PRIu64 " threads=%2d  %8.3f s  %9.0f reads/s  %6.2f GiB/s  bad=%" PRIu64 "%s\n",
                mode.c_str(), size, h.n, threads, dt, h.n / dt, (double) h.bytes / dt / (1024 * 1024 * 1024), (uint64_t) bad.load(),
                drop ? "  (cold)" : "");
    if (map) ::munmap(map, (size_t) fsize);
    ::close(fd);
    return h;
}

// ---------------------------------------------------------------- module bench

static void run_module(const std::string & gguf, const std::string & dir, const std::string & seq, std::vector<int> toks,
                       int threads, int reps, int check_n, bool cold, int split) {
    GgufModel m = GgufModel::open(gguf);
    Ds41Geometry G;
    std::string e = ds41_read_geometry(m, G);
    if (!e.empty()) { std::fprintf(stderr, "bench: geometry: %s\n", e.c_str()); std::exit(1); }
    Ds41Engram eng;
    std::vector<std::string> dirs;   // ':'-separated, as ds41_generate --engram
    for (size_t a = 0, b; a <= dir.size(); a = b + 1) { b = dir.find(':', a); if (b == std::string::npos) b = dir.size(); if (b > a) dirs.push_back(dir.substr(a, b - a)); }
    if (!eng.open(G, dirs, threads, e)) { std::fprintf(stderr, "bench: engram open: %s\n", e.c_str()); std::exit(1); }
    std::printf("engram: layers=%zu heads=%" PRId64 " dim=%" PRId64 " ngram=%" PRId64 " cols=%" PRId64 " rows=[",
                G.engram_layers.size(), G.engram_heads, G.engram_dim, G.engram_ngram, eng.n_cols());
    for (size_t i = 0; i < G.engram_rows.size(); ++i) std::printf("%s%" PRId64, i ? "," : "", G.engram_rows[i]);
    std::printf("]  threads=%d\n", threads);

    auto seqv = load_i32(seq);
    if (seqv.empty()) { std::fprintf(stderr, "bench: empty seq\n"); std::exit(2); }
    int64_t maxn = 0;
    for (int t : toks) maxn = std::max<int64_t>(maxn, t);
    if ((int64_t) seqv.size() < maxn) { std::fprintf(stderr, "bench: seq has %zu tokens, need %" PRId64 "\n", seqv.size(), maxn); std::exit(2); }

    // tokens are recorded by position exactly as ds41_dense does before a pass
    for (int64_t p = 0; p < (int64_t) seqv.size(); ++p) eng.set_token(p, seqv[(size_t) p]);

    std::vector<std::vector<float>> a, b;
    const int64_t cols = eng.n_cols(), dim = eng.row_dim(), nel = (int64_t) G.engram_layers.size();

    std::printf("\n%-6s %-10s %10s %12s %12s %12s %8s\n", "tokens", "path", "wall ms", "rows/s", "syscalls", "sysc/s", "speedup");
    for (int n : toks) {
        double best_ref = 1e30, best_new = 1e30;
        uint64_t nref = 0, nnew = 0;
        for (int r = 0; r < reps; ++r) {
            if (cold) eng.drop_cache();
            uint64_t b0 = eng.bytes_read();
            double t0 = now_s();
            if (!eng.rows_reference(0, n, a, e)) { std::fprintf(stderr, "bench: rows_reference: %s\n", e.c_str()); std::exit(1); }
            double dt = now_s() - t0;
            if (dt < best_ref) { best_ref = dt; nref = (eng.bytes_read() - b0) / (uint64_t) (dim + dim / 32); }
            if (cold) eng.drop_cache();
            uint64_t r0 = eng.row_reads();
            t0 = now_s();
            if (!eng.rows(0, n, b, e)) { std::fprintf(stderr, "bench: rows: %s\n", e.c_str()); std::exit(1); }
            dt = now_s() - t0;
            if (dt < best_new) { best_new = dt; nnew = eng.row_reads() - r0; }
            if (split) {   // new path only: how much is hash+sort+fire vs the join
                const double t0b = now_s();
                auto bh = eng.start(0, n, b, e);
                const double t1 = now_s();
                if (!eng.wait(bh, e)) std::exit(1);
                const double t2 = now_s();
                std::printf("  split n=%-5d hash+sort+fire %6.2f ms   wait %6.2f ms\n", n, (t1 - t0b) * 1e3, (t2 - t1) * 1e3);
            }
        }
        const double rows = (double) nel * n * cols;
        // the reference path issues two preads (weight + scale) per row; the fast path reports its own count
        const uint64_t sref = (uint64_t) nel * n * cols * 2;
        std::printf("%-6d %-10s %10.2f %12.0f %12" PRIu64 " %12.0f %8.1f\n", n, "old", best_ref * 1e3, rows / best_ref,
                    sref, sref / best_ref, 1.0);
        std::printf("%-6d %-10s %10.2f %12.0f %12" PRIu64 " %12.0f %8.1f\n", n, "new", best_new * 1e3, rows / best_new,
                    nnew, nnew / best_new, best_ref / best_new);
    }

    if (check_n > 0) {
        // byte-identity: the fast path must reproduce rows_reference exactly, over real tokens
        int nbad = 0;
        std::vector<std::vector<float>> x, y;
        if (!eng.rows_reference(0, check_n, x, e) || !eng.rows(0, check_n, y, e)) { std::fprintf(stderr, "bench: check: %s\n", e.c_str()); std::exit(1); }
        for (size_t s = 0; s < x.size(); ++s) {
            if (x[s].size() != y[s].size()) { nbad++; continue; }
            if (std::memcmp(x[s].data(), y[s].data(), x[s].size() * sizeof(float)) != 0) nbad++;
        }
        std::printf("\nCHECK %d tokens: old vs new %s (%d/%zu slots differ)\n", check_n, nbad ? "MISMATCH" : "identical", nbad, x.size());
        if (nbad) std::exit(1);
    }

    // async API: start now, wait later - same bytes, and it must overlap
    if (check_n > 0) {
        std::vector<std::vector<float>> z;
        auto bh = eng.start(0, check_n, z, e);
        double t0 = now_s();
        if (!eng.wait(bh, e)) { std::fprintf(stderr, "bench: wait: %s\n", e.c_str()); std::exit(1); }
        double dt = now_s() - t0;
        int nbad = 0;
        std::vector<std::vector<float>> x;
        if (!eng.rows_reference(0, check_n, x, e)) std::exit(1);
        for (size_t s = 0; s < x.size(); ++s) if (std::memcmp(x[s].data(), z[s].data(), x[s].size() * sizeof(float)) != 0) nbad++;
        std::printf("ASYNC %d tokens: start+wait %.2f ms, %s\n", check_n, dt * 1e3, nbad ? "MISMATCH" : "identical");
    }
}

int main(int argc, char ** argv) {
    std::string gguf, dir, seq, raw_file, raw_mode = "pread";
    std::vector<int> toks;
    int threads = 16, reps = 3, check_n = 0;
    bool cold = false;
    int split = 0;
    int64_t raw_n = 20000, raw_size = 4096;
    std::vector<int> raw_threads_v = { 1, 4, 8, 16, 32 };
    bool raw_drop = false;
    uint64_t raw_lo = 0, raw_hi = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nxt = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (k == "--gguf") gguf = nxt();
        else if (k == "--dir") dir = nxt();
        else if (k == "--seq") seq = nxt();
        else if (k == "--threads") threads = std::atoi(nxt().c_str());
        else if (k == "--reps") reps = std::atoi(nxt().c_str());
        else if (k == "--tokens") { std::string s = nxt(); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); toks.push_back(std::atoi(s.substr(p, q - p).c_str())); if (q == std::string::npos) break; p = q + 1; } }
        else if (k == "--check") check_n = std::atoi(nxt().c_str());
        else if (k == "--cold") cold = true;
        else if (k == "--split") split = 1;
        else if (k == "--raw") raw_file = nxt();
        else if (k == "--raw-n") raw_n = std::atoll(nxt().c_str());
        else if (k == "--raw-threads") { std::string s = nxt(); raw_threads_v.clear(); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); raw_threads_v.push_back(std::atoi(s.substr(p, q - p).c_str())); if (q == std::string::npos) break; p = q + 1; } }
        else if (k == "--raw-size") raw_size = std::atoll(nxt().c_str());
        else if (k == "--raw-drop") raw_drop = true;
        else if (k == "--raw-mode") raw_mode = nxt();
        else if (k == "--raw-lo") raw_lo = std::strtoull(nxt().c_str(), nullptr, 0);
        else if (k == "--raw-hi") raw_hi = std::strtoull(nxt().c_str(), nullptr, 0);
        else { std::fprintf(stderr, "bench: unknown arg %s\n", k.c_str()); return 2; }
    }
    if (!raw_file.empty()) {
        struct stat st;
        if (::stat(raw_file.c_str(), &st) != 0) { std::fprintf(stderr, "bench: stat %s failed\n", raw_file.c_str()); return 2; }
        if (raw_hi == 0) raw_hi = (uint64_t) st.st_size;
        std::printf("file %s  size %.1f GiB  window [%" PRIu64 ", %" PRIu64 ")%.1f GiB%s\n", raw_file.c_str(),
                    (double) st.st_size / (1024.0 * 1024 * 1024), raw_lo, raw_hi,
                    (double) (raw_hi - raw_lo) / (1024.0 * 1024 * 1024), raw_drop ? "  cold" : "  warm");
        for (int t : raw_threads_v) raw_bench(raw_file, raw_n, t, raw_mode, raw_lo, raw_hi, raw_size, raw_drop);
        return 0;
    }
    if (toks.empty()) toks = { 1, 512, 1024, 4096 };
    if (gguf.empty() || dir.empty() || seq.empty()) {
        std::fprintf(stderr, "usage: bench_ds41_engram --gguf G --dir D --seq S [--tokens 1,512,1024,4096] [--threads 16] [--reps 3] [--check N]\n"
                             "       bench_ds41_engram --raw FILE [--raw-n N] [--raw-threads T] [--raw-mode pread|odirect|mmap]\n");
        return 2;
    }
    run_module(gguf, dir, seq, toks, threads, reps, check_n, cold, split);
    return 0;
}
