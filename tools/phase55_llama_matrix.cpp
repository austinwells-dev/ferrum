// Persistent reference runner built against current upstream llama.cpp's C API.
// Requests supply token IDs directly so prompt tokenization is identical to Ferrum.
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

static double milliseconds(clock_type::duration elapsed) {
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

static llama_token argmax(const float * logits, int32_t vocab) {
    if (logits == nullptr || vocab <= 0) {
        throw std::runtime_error("llama.cpp returned no logits");
    }
    int32_t best = 0;
    for (int32_t i = 1; i < vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static size_t process_max_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    // macOS reports ru_maxrss in bytes (Linux reports KiB).
    return static_cast<size_t>(usage.ru_maxrss);
}

static void check_decode(llama_context * ctx, llama_batch batch) {
    const int32_t status = llama_decode(ctx, batch);
    if (status != 0) {
        throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
    }
    llama_synchronize(ctx);
}

static json run_request(
    llama_context * ctx,
    const llama_model * model,
    const json & request,
    uint32_t n_ctx,
    enum ggml_type kv_type) {
    const std::string case_name = request.at("case").get<std::string>();
    const size_t pair = request.at("pair").get<size_t>();
    const std::string pair_order = request.at("pair_order").get<std::string>();
    const bool warmup = request.value("warmup", false);
    const int max_new_tokens = request.at("max_new_tokens").get<int>();
    std::vector<llama_token> prompt = request.at("prompt_ids").get<std::vector<llama_token>>();
    if (prompt.empty() || max_new_tokens <= 0) {
        throw std::runtime_error("prompt_ids and max_new_tokens must be nonempty/positive");
    }
    if (prompt.size() + static_cast<size_t>(max_new_tokens) > n_ctx) {
        throw std::runtime_error("request exceeds the fixed llama.cpp context capacity");
    }

    // Clearing the sequence is outside generation timing; the context and its KV allocation
    // persist across requests, matching a warmed inference session.
    llama_memory_clear(llama_get_memory(ctx), false);
    const auto generation_start = clock_type::now();
    const auto prefill_start = generation_start;
    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    check_decode(ctx, batch);
    const double prefill_ms = milliseconds(clock_type::now() - prefill_start);

    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto sample_start = clock_type::now();
    llama_token token = argmax(llama_get_logits_ith(ctx, -1), vocab);
    double sampling_ms = milliseconds(clock_type::now() - sample_start);
    std::vector<llama_token> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(token);
    const double first_token_ms = milliseconds(clock_type::now() - generation_start);
    std::vector<double> decode_ms;
    decode_ms.reserve(static_cast<size_t>(max_new_tokens - 1));

    for (int step = 1; step < max_new_tokens; ++step) {
        const auto decode_start = clock_type::now();
        batch = llama_batch_get_one(&token, 1);
        check_decode(ctx, batch);
        decode_ms.push_back(milliseconds(clock_type::now() - decode_start));
        const auto next_sample_start = clock_type::now();
        token = argmax(llama_get_logits_ith(ctx, -1), vocab);
        sampling_ms += milliseconds(clock_type::now() - next_sample_start);
        generated.push_back(token);
    }
    const double generation_ms = milliseconds(clock_type::now() - generation_start);
    std::vector<double> sorted_decode = decode_ms;
    std::sort(sorted_decode.begin(), sorted_decode.end());
    const double median_decode_ms = sorted_decode.empty()
        ? 0.0
        : sorted_decode[sorted_decode.size() / 2];
    const double decode_sum_ms = std::accumulate(decode_ms.begin(), decode_ms.end(), 0.0);

    return json{
        {"runtime", "llama.cpp"},
        {"llama_version", llama_version()},
        {"case", case_name},
        {"pair", pair},
        {"pair_order", pair_order},
        {"warmup", warmup},
        {"model_format", "GGUF"},
        {"prompt_tokens", prompt.size()},
        {"generated_tokens", generated.size()},
        {"prompt_ids", prompt},
        {"generated_ids", generated},
        {"generation_ms", generation_ms},
        {"generation_tps", generated.size() * 1000.0 / generation_ms},
        {"prefill_ms", prefill_ms},
        {"prefill_tps", prompt.size() * 1000.0 / prefill_ms},
        {"first_token_ms", first_token_ms},
        {"decode_ms", decode_ms},
        {"decode_median_ms", median_decode_ms},
        {"decode_tps", median_decode_ms > 0.0 ? 1000.0 / median_decode_ms : 0.0},
        {"decode_aggregate_tps", decode_sum_ms > 0.0 ? decode_ms.size() * 1000.0 / decode_sum_ms : 0.0},
        {"sampling_ms", sampling_ms},
        {"kv_type", ggml_type_name(kv_type)},
        {"kv_context_capacity_tokens", n_ctx},
        {"command_buffer_count", nullptr},
        {"dispatch_count", nullptr},
        {"process_max_rss_bytes", process_max_rss_bytes()},
    };
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: phase55_llama_matrix MODEL.gguf\n";
        return 2;
    }
    try {
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (model == nullptr) {
            throw std::runtime_error("could not load model");
        }

        constexpr uint32_t n_ctx = 4096;
        constexpr uint32_t n_batch = 2048;
        constexpr uint32_t n_ubatch = 512;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = n_ctx;
        context_params.n_batch = n_batch;
        context_params.n_ubatch = n_ubatch;
        context_params.n_seq_max = 1;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        context_params.type_k = GGML_TYPE_BF16;
        context_params.type_v = GGML_TYPE_BF16;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.no_perf = false;
        llama_context * ctx = llama_init_from_model(model, context_params);
        if (ctx == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("could not create llama context");
        }
        std::cerr << "phase55_llama_matrix ready: version=" << llama_version()
                  << " n_ctx=" << llama_n_ctx(ctx)
                  << " n_batch=" << llama_n_batch(ctx)
                  << " n_ubatch=" << llama_n_ubatch(ctx)
                  << " kv=bf16 flash_attention=enabled gpu_layers=99\n";

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const json request = json::parse(line);
                const json response = run_request(ctx, model, request, n_ctx, GGML_TYPE_BF16);
                std::cout << response.dump() << std::endl;
            } catch (const std::exception & error) {
                std::cout << json{{"error", error.what()}}.dump() << std::endl;
            }
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << "phase55_llama_matrix: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
