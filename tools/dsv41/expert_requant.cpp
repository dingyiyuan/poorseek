// expert_requant - requantize a dsv41 pack's routed experts into ggml types the port's int8 DPAS expert kernels take
// (gate/up -> IQ2_S, down -> IQ4_NL), through ggml's own decoders and quantizer, into one override file plus a JSON the
// pack patcher reads.  Nothing in the pack is touched: the caller adds a `file` line and repoints the tensors.
//
//   expert_requant PACK OUT.bin OUT.json [--layers 0-7] [--gu IQ2_S] [--down IQ4_NL] [--jobs N]
//
// PACK is a dsv41 pack directory (ds4.txt + tensors.txt + the model's shards).  The source tensors are the pack's own
// bytes (any ggml type ggml can decode), so this works from the pack rather than from the original checkpoint.
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Tensor {
    std::string name;
    std::string type;
    int file = 0;
    int64_t off = 0, bytes = 0;
    int64_t rows = 0, cols = 0;   // dims outermost first
};

struct File {
    std::string path;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t i = 0;
    while (i < s.size()) {
        auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
        while (i < s.size() && ws(s[i])) ++i;
        size_t j = i;
        while (j < s.size() && !ws(s[j])) ++j;
        if (j > i) v.push_back(s.substr(i, j - i));
        i = j;
    }
    return v;
}

bool ieq(const std::string& a, const char* b) {   // this ggml spells its names lower case ("iq2_s")
    size_t i = 0;
    for (; i < a.size() && b[i]; ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = (char) (x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = (char) (y - 'A' + 'a');
        if (x != y) return false;
    }
    return i == a.size() && b[i] == 0;
}
int type_id(const std::string& n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t)
        if (ieq(n, ggml_type_name((enum ggml_type) t))) return t;
    return -1;
}

size_t row_bytes(int t, int64_t cols) { return ggml_row_size((enum ggml_type) t, cols); }

bool parse_range(const std::string& s, std::set<int>& out) {
    size_t i = 0;
    while (i < s.size()) {
        int a = 0, b = 0;
        if (std::sscanf(s.c_str() + i, "%d-%d", &a, &b) == 2) {
            for (int k = a; k <= b; ++k) out.insert(k);
            i = s.find(',', i);
        } else if (std::sscanf(s.c_str() + i, "%d", &a) == 1) {
            out.insert(a);
            i = s.find(',', i);
        } else {
            return false;
        }
        if (i == std::string::npos) break;
        ++i;
    }
    return true;
}

// layer of "layers.<L>.ffn.experts.<E>.w1.weight"
int layer_of(const std::string& n) {
    const std::string p = "layers.";
    if (n.compare(0, p.size(), p) != 0) return -1;
    return std::atoi(n.c_str() + p.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: expert_requant PACK OUT.bin OUT.json [--layers 0-7] [--gu IQ2_S] [--down IQ4_NL] [--jobs N]\n");
        return 2;
    }
    const std::string pack = argv[1], out_bin = argv[2], out_json = argv[3];
    std::string gu_name = "IQ2_S", down_name = "IQ4_NL", layers_s = "0-7";
    int jobs = 6;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--gu") gu_name = argv[i + 1];
        else if (k == "--down") down_name = argv[i + 1];
        else if (k == "--layers") layers_s = argv[i + 1];
        else if (k == "--jobs") jobs = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    std::set<int> layers;
    if (!parse_range(layers_s, layers)) { std::fprintf(stderr, "bad --layers\n"); return 2; }
    const int tgu = type_id(gu_name), tdn = type_id(down_name);
    if (tgu < 0 || tdn < 0) { std::fprintf(stderr, "unknown type\n"); return 2; }
    // (ggml_quantize_chunk is the encoder: this ggml's traits carry only the decoders)

    // the pack
    std::vector<File> files;
    std::vector<Tensor> tens;
    {
        FILE* f = std::fopen((pack + "/tensors.txt").c_str(), "r");
        if (!f) { std::fprintf(stderr, "no tensors.txt in %s\n", pack.c_str()); return 2; }
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            auto v = split(line);
            if (v.empty()) continue;
            if (v[0] == "file" && v.size() >= 3) {
                const int idx = std::atoi(v[1].c_str());
                if ((int) files.size() <= idx) files.resize(idx + 1);
                files[idx].path = v[2];
                continue;
            }
            if (v[0] != "t" || v.size() < 7) continue;
            Tensor t;
            t.name = v[1];
            t.type = v[2];
            t.file = std::atoi(v[3].c_str());
            t.off = std::atoll(v[4].c_str());
            t.bytes = std::atoll(v[5].c_str());
            t.rows = std::atoll(v[6].c_str());
            t.cols = v.size() > 7 ? std::atoll(v[7].c_str()) : 1;
            tens.push_back(t);
        }
        std::fclose(f);
    }
    // relative file names resolve against ds4.txt's model_dir (the shards live with the model, not in the pack)
    std::string model_dir = pack;
    {
        FILE* d = std::fopen((pack + "/ds4.txt").c_str(), "r");
        if (d) {
            char line[4096];
            while (std::fgets(line, sizeof line, d)) {
                auto v = split(line);
                if (v.size() >= 2 && (v[0] == "model_dir" || v[0] == "model-dir")) { model_dir = v[1]; break; }
                if (v.size() >= 2 && v[0] == "dir") { model_dir = v[1]; break; }
            }
            std::fclose(d);
        }
    }
    for (auto& f : files)
        if (!f.path.empty() && f.path[0] != '/') f.path = model_dir + "/" + f.path;

    struct Job {
        const Tensor* t;
        int src, dst;
    };
    std::vector<Job> todo;
    for (const auto& t : tens) {
        const int L = layer_of(t.name);
        if (L < 0 || !layers.count(L)) continue;
        const bool up = t.name.find(".ffn.experts.") != std::string::npos;
        if (!up) continue;
        const bool down = t.name.find(".w2.") != std::string::npos;
        const int src = type_id(t.type);
        if (src < 0) continue;
        const ggml_type_traits* ts = ggml_get_type_traits((enum ggml_type) src);
        if (!ts || !ts->to_float) continue;
        const int dst = down ? tdn : tgu;
        if (src == dst) continue;              // already the target type
        Job j{&t, src, dst};
        // the shape ggml_quantize_chunk needs
        if (t.rows <= 0 || t.cols <= 0 || t.cols % ggml_blck_size((enum ggml_type) dst) != 0) continue;
        todo.push_back(j);
    }
    if (todo.empty()) { std::fprintf(stderr, "nothing to convert\n"); return 1; }
    std::printf("%zu tensors of layers %s -> %s (gate/up) and %s (down)\n", todo.size(), layers_s.c_str(), gu_name.c_str(),
                down_name.c_str());

    const int fd_out = ::open(out_bin.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd_out < 0) { std::perror(out_bin.c_str()); return 2; }
    std::vector<std::pair<const Tensor*, int64_t>> placed(todo.size());   // tensor -> offset in out.bin
    std::atomic<size_t> next{0}, done{0};
    std::mutex mu;
    std::vector<std::string> errors;
    int64_t at = 0;
    for (size_t i = 0; i < todo.size(); ++i) {   // reserve offsets up front (alignment 4096)
        const Tensor& t = *todo[i].t;
        const int dst = todo[i].dst;
        const size_t rb = row_bytes(dst, t.cols);
        const int64_t bytes = (int64_t) rb * t.rows;
        placed[i] = {&t, at};
        at = (at + bytes + 4095) / 4096 * 4096;
    }
    if (::ftruncate(fd_out, at) != 0) { std::perror("ftruncate"); return 2; }

    auto work = [&]() {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= todo.size()) break;
            const Tensor& t = *todo[i].t;
            const int src = todo[i].src, dst = todo[i].dst;
            const ggml_type_traits* ts = ggml_get_type_traits((enum ggml_type) src);
            const size_t srb = row_bytes(src, t.cols), drb = row_bytes(dst, t.cols);
            const int fd_in = ::open(files[(size_t) t.file].path.c_str(), O_RDONLY);
            if (fd_in < 0) {
                std::lock_guard<std::mutex> lk(mu);
                errors.push_back("open " + files[(size_t) t.file].path);
                continue;
            }
            // one read a tensor: a row at a time made the syscalls the bottleneck (5 MB/s against 55)
            std::vector<uint8_t> srcbuf((size_t) srb * (size_t) t.rows), dstbuf((size_t) drb * (size_t) t.rows);
            std::vector<float> fr((size_t) t.cols);
            int64_t wrote = 0;
            if (::pread(fd_in, srcbuf.data(), (size_t) srb * (size_t) t.rows, t.off) !=
                (ssize_t) ((size_t) srb * (size_t) t.rows)) {
                std::lock_guard<std::mutex> lk(mu);
                errors.push_back("short read " + t.name);
                ::close(fd_in);
                continue;
            }
            for (int64_t r = 0; r < t.rows; ++r) {
                const uint8_t* srcrow_p = srcbuf.data() + (size_t) r * srb;
                uint8_t* dstrow_p = dstbuf.data() + (size_t) r * drb;
                ts->to_float(srcrow_p, fr.data(), t.cols);
                if (ggml_quantize_chunk((enum ggml_type) dst, fr.data(), dstrow_p, 0, 1, t.cols, nullptr) != drb) {
                    std::lock_guard<std::mutex> lk(mu);
                    errors.push_back("quantize " + t.name);
                    break;
                }
                wrote += (int64_t) drb;
            }
            if (::pwrite(fd_out, dstbuf.data(), (size_t) drb * (size_t) t.rows, placed[i].second) !=
                (ssize_t) ((size_t) drb * (size_t) t.rows)) {
                std::lock_guard<std::mutex> lk(mu);
                errors.push_back("short write " + t.name);
            } else {
                // (keep our own 176 GB of output out of the page cache so it does not evict the input shards)
                ::posix_fadvise(fd_out, (off_t) placed[i].second, (off_t) ((size_t) drb * (size_t) t.rows),
                                POSIX_FADV_DONTNEED);
            }
            ::close(fd_in);
            const size_t n = done.fetch_add(1) + 1;
            if (n % 32 == 0 || n == todo.size())
                std::printf("  %zu/%zu tensors (%" PRId64 " MB written)\n", n, todo.size(), wrote / (1 << 20));
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < std::max(1, jobs); ++i) th.emplace_back(work);
    for (auto& x : th) x.join();
    ::close(fd_out);
    if (!errors.empty()) {
        std::fprintf(stderr, "%zu errors, first: %s\n", errors.size(), errors.front().c_str());
        return 1;
    }

    // the JSON the pack patcher reads: name -> [type, file, offset, bytes, dims...]
    FILE* j = std::fopen(out_json.c_str(), "w");
    if (!j) { std::perror(out_json.c_str()); return 2; }
    std::fprintf(j, "{\n");
    for (size_t i = 0; i < todo.size(); ++i) {
        const Tensor& t = *todo[i].t;
        const int dst = todo[i].dst;
        const int64_t bytes = (int64_t) row_bytes(dst, t.cols) * t.rows;
        std::fprintf(j, "  \"%s\": [\"%s\", %" PRId64 ", %" PRId64 ", %" PRId64 "%s]\n%s", t.name.c_str(),
                     ggml_type_name((enum ggml_type) dst), placed[i].second, bytes, t.rows,
                     t.cols > 0 ? "" : "", i + 1 == todo.size() ? "" : ",");
        (void) t.off;
    }
    std::fprintf(j, "}\n");
    std::fclose(j);
    std::printf("wrote %s (%" PRId64 " MB) and %s\n", out_bin.c_str(), at >> 20, out_json.c_str());
    return 0;
}
