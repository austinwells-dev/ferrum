// Dump llama.cpp's LFM3-Audio encoder output for raw 16 kHz f32 samples.
// usage: embdump LANG.gguf MMPROJ.gguf SAMPLES.f32 OUT.f32
#include "llama.h"
#include "mtmd.h"
#include <cstdio>
#include <vector>
#include <string>
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: embdump LANG MMPROJ SAMPLES OUT\n"); return 2; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = mtmd_context_params_default();
    cp.use_gpu = true;
    cp.warmup = false;
    mtmd_context * ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) return 1;
    FILE * f = fopen(argv[3], "rb");
    std::vector<float> s;
    float v;
    while (fread(&v, 4, 1, f) == 1) s.push_back(v);
    fclose(f);
    mtmd_bitmap * bm = mtmd_bitmap_init_from_audio(s.size(), s.data());
    std::string text = mtmd_default_marker();
    mtmd_input_text in = {text.c_str(), text.size(), false, true};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bms[1] = {bm};
    if (mtmd_tokenize(ctx, chunks, &in, bms, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            size_t n = 0;
            mtmd_input_chunk_get_tokens_text(c, &n);
            fprintf(stderr, "text chunk of %zu tokens\n", n);
            continue;
        }
        if (mtmd_encode_chunk(ctx, c) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        size_t n = mtmd_input_chunk_get_n_tokens(c);
        int w = llama_model_n_embd_inp(model);
        float * e = mtmd_get_output_embd(ctx);
        FILE * o = fopen(argv[4], "wb");
        fwrite(e, 4, n * w, o);
        fclose(o);
        fprintf(stderr, "audio chunk: %zu rows of %d\n", n, w);
    }
    return 0;
}
