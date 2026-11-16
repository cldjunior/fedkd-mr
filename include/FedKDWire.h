#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Bloco 6: envelope binario fixo de 16 bytes, ESP32 little-endian.
// Nao e autenticacao criptografica: apenas impede misturas acidentais
// de mensagens de diferentes experimentos/rodadas.
namespace FedKDWire {
static constexpr size_t HEADER_LEN = 16;
static constexpr uint8_t MAGIC[4] = {'F','K','D','6'};
static inline uint64_t hashExperiment(const char* s) {
    uint64_t h = UINT64_C(14695981039346656037);
    if (!s) return h;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        h ^= (uint64_t)(*p);
        h *= UINT64_C(1099511628211);
    }
    return h;
}
static inline void write32(uint8_t* p, uint32_t n) {
    for (unsigned i=0;i<4;++i) p[i] = (uint8_t)(n >> (8*i));
}
static inline void write64(uint8_t* p, uint64_t n) {
    for (unsigned i=0;i<8;++i) p[i] = (uint8_t)(n >> (8*i));
}
static inline uint32_t read32(const uint8_t* p) {
    uint32_t n=0;
    for (unsigned i=0;i<4;++i) n |= (uint32_t)p[i] << (8*i);
    return n;
}
static inline uint64_t read64(const uint8_t* p) {
    uint64_t n=0;
    for (unsigned i=0;i<8;++i) n |= (uint64_t)p[i] << (8*i);
    return n;
}
static inline void build(uint8_t* p, uint32_t round, const char* experiment_id) {
    memcpy(p, MAGIC, 4);
    write32(p+4, round);
    write64(p+8, hashExperiment(experiment_id));
}
static inline bool valid(const uint8_t* p, size_t length, uint32_t round, const char* experiment_id) {
    return length>=HEADER_LEN && memcmp(p, MAGIC, 4)==0 && read32(p+4)==round
        && read64(p+8)==hashExperiment(experiment_id);
}
}
