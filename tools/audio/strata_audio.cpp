// tools/audio/strata_audio.cpp - the audio half of Strata's multimodal path (local, Indra's Mirror).
//
// Turns an audio clip into the embeddings the ASR decoder reads in place of its <|audio_pad|> tokens, with
// llama.cpp's mtmd library and the model's audio mmproj (audio encoder + projector).  The engine does the rest:
// it places the rows at the audio's pad tokens and gives them their positions - see --serve GENI and the audio
// record tag in src/program/generate.cpp.  This is the twin of tools/vision/strata_vision.cpp.
//
//   strata-audio --mmproj <mmproj.gguf> --model <asr decoder .gguf, first split> [--gpu] [--threads N]
//                [--max-tokens N] [--flash-attn on|off|auto]
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <audio path> <output path>   ->  "OK <n_tokens> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31415653 'SVA1', n_tokens, n_tokens, 1, n_embd}  then float32 [n_tokens][n_embd].
// The 'SVA1' tag (vs the vision tool's 'SVE1') marks an AUDIO record: the engine places the rows at consecutive
// positions p, p+1, ... (1-D), not a 2-D image grid.
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <audio> <out>": paths may contain spaces, so the audio path ends at the last " " before the output path
bool parse_enc(const std::string& line, std::string& audio, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    audio = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !audio.empty() && !out.empty();
}

// The projector's output width is the text model's input width.  Audio projectors carry clip.audio.projection_dim;
// fall back to the vision one so a combined projector still reports a width.
int projector_dim(gguf_context* gg) {
    for (const char* key : {"clip.audio.projection_dim", "clip.vision.projection_dim"}) {
        const int64_t k = gguf_find_key(gg, key);
        if (k >= 0) return (int) gguf_get_val_u32(gg, k);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model;
    bool gpu = false;
    int threads = 0, max_tokens = 0;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--flash-attn") {
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-audio --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
                             "[--max-tokens N] [--flash-attn on|off|auto]\n");
        return 2;
    }
    // On the CPU keep the GPU unseen (same reason as strata-vision: a CUDA build would otherwise open a context
    // there and take VRAM from the engine beside it).
    if (!gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();

#if !defined(_WIN32)
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;                                // the text model is opened vocab-only (no weights)
    llama_model* text = llama_model_load_from_file(model.c_str(), mp);
    if (!text) { std::printf("ERR cannot open the text model %s\n", model.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    // Same AVX-512 head-size trap as the vision encoder: on the CPU, "auto" can turn on a tiled kernel that is
    // slower here, so a CPU run without an explicit --flash-attn turns it off on an AVX-512 host.
    if (!gpu && !fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    if (threads <= 0 && !gpu) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;   // reuse: for audio this caps encoder tokens
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_audio(ctx)) {
        std::printf("ERR cannot load the audio encoder %s (no audio support)\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    const int sample_rate = mtmd_get_audio_sample_rate(ctx);

    int n_embd = 0;
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        if (gg) { n_embd = projector_dim(gg); gguf_free(gg); }
    }
    if (n_embd <= 0) { std::printf("ERR the audio encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    std::fprintf(stderr, "strata-audio: audio encoder ready (%d Hz, %d embd)\n", sample_rate, n_embd);
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    const std::string marker = mtmd_default_marker();
    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string audio, out;
        if (!parse_enc(line, audio, out)) { std::printf("ERR expected: ENC <audio> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        // the helper auto-detects the file type from its magic bytes (wav / mp3 / flac)
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, audio.c_str(), false,
                                                                          mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::printf("ERR cannot read the audio %s\n", audio.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the audio could not be preprocessed";
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no audio chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the audio encoder failed";
        if (err.empty()) {
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = std::fopen(out.c_str(), "wb");
            if (!f) err = "cannot write " + out;
            if (f) {
                // nx = n, ny = 1: one row, so nx*ny == n; the 'SVA1' tag is what makes the engine read it as audio
                const int32_t hdr[5] = {0x31415653, n, n, 1, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                if (ok) {
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::printf("OK %d %.0f\n", n, ms);
                }
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
