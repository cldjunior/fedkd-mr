#pragma once
// FedKD-MR — checkpoint experimental opt-in; two slots and verified FNV-1a checksum.
// Requires Model.h, Config.h and LittleFS.h included before this header.
#include <stdint.h>
#include <string.h>

// Identidade definida pelo comando MQTT start; nao fica embutida no firmware.
static char g_fedkd_experiment_id[65] = {};

namespace FedKDCheckpoint {
static bool setExperimentId(const char* id) {
    if (!id) return false;
    size_t n = strlen(id);
    if (n < 1 || n >= sizeof(g_fedkd_experiment_id)) return false;
    for (size_t i=0;i<n;++i) {
        char c=id[i];
        if (!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_')) return false;
    }
    if (g_fedkd_experiment_id[0] && strcmp(id,g_fedkd_experiment_id)!=0) return false;
    memcpy(g_fedkd_experiment_id,id,n+1);
    return true;
}
static const char* experimentId() {return g_fedkd_experiment_id;}

static constexpr uint32_t MAGIC = 0x464B4331u; // FKC1
static constexpr uint32_t VERSION = 1;
static const char* const SLOT_A = "/fedkd_ckpt_a.bin";
static const char* const SLOT_B = "/fedkd_ckpt_b.bin";
static const char* const TMP = "/fedkd_ckpt_write.tmp";

struct Header {
    uint32_t magic, version, client_id, model_arch;
    uint32_t experiment_hash, completed_round, payload_bytes, checksum;
};

static uint32_t fnv(const uint8_t* p, size_t n, uint32_t h=2166136261UL) {
    for (size_t i=0;i<n;++i) { h ^= p[i]; h *= 16777619UL; }
    return h;
}
static uint32_t experimentHash() {
    const char* s=experimentId();
    return fnv((const uint8_t*)s, strlen(s));
}
static bool enabled() { return experimentId()[0] != '\0'; }
static uint32_t modelBytes() {
#if MODEL_ARCH == MODEL_ARCH_MLP
    return 25612u; // MLP2, 96-64-4; SINGLE_BIAS_PER_LAYER
#elif MODEL_ARCH == MODEL_ARCH_CNN1D
    return 30612u;
#elif MODEL_ARCH == MODEL_ARCH_GRU
    return 60180u;
#else
    return 0;
#endif
}
static bool checksum(File& f, uint32_t length, uint32_t& out) {
    uint8_t buf[256];
    uint32_t h=2166136261UL, left=length;
    while(left) {
        const size_t n=left > sizeof(buf) ? sizeof(buf) : left;
        if(f.read(buf,n)!=n) return false;
        h=fnv(buf,n,h); left-=(uint32_t)n;
    }
    out=h; return true;
}
static bool verify(const char* path, Header& h) {
    if (!LittleFS.exists(path)) return false;
    File f=LittleFS.open(path,"r");
    if (!f) return false;
    const bool header_ok=f.read((uint8_t*)&h,sizeof(h)) == sizeof(h);
    bool ok=header_ok && h.magic==MAGIC && h.version==VERSION &&
        h.client_id==(uint32_t)CLIENT_ID && h.model_arch==(uint32_t)MODEL_ARCH &&
        h.experiment_hash==experimentHash() && h.completed_round>0 &&
        h.payload_bytes==modelBytes() && f.size()==sizeof(h)+(size_t)h.payload_bytes;
    if(ok) { uint32_t c=0; ok=checksum(f,h.payload_bytes,c) && c==h.checksum; }
    f.close(); return ok;
}
// Result: 1=restored; 0=fresh round one; -1=invalid/nonsequential checkpoint.
static int restoreForRound(Model& m, int round, uint32_t* restored_hash=nullptr) {
    if (!enabled()) return -1; // Identidade obrigatoria para recuperar.
    Header a={},b={};
    bool va=verify(SLOT_A,a), vb=verify(SLOT_B,b);
    const char* selected=nullptr;
    Header h={};
    if(va && (!vb || a.completed_round>=b.completed_round)) { selected=SLOT_A; h=a; }
    else if(vb) { selected=SLOT_B; h=b; }
    if(round<=0) return -1;
    if(!selected) {
        // A corrupted checkpoint for THIS experiment must not cause a silent fresh start.
        const char* paths[]={SLOT_A,SLOT_B};
        for(const char* path:paths) {
            if(!LittleFS.exists(path)) continue;
            File r=LittleFS.open(path,"r");
            Header raw={};
            const bool seen=r && r.read((uint8_t*)&raw,sizeof(raw))==sizeof(raw);
            if(r) r.close();
            if(seen && raw.magic==MAGIC && raw.experiment_hash==experimentHash()) {
                Serial.println("[CKPT] Checkpoint deste experimento existe mas falhou na integridade.");
                return -1;
            }
        }
        if(round==1) return 0;
        Serial.println("[CKPT] Sem checkpoint valido: impossivel retomar rodada posterior.");
        return -1;
    }
    if(round==1 || (uint32_t)round!=h.completed_round+1) {
        Serial.printf("[CKPT] RODADA INCOMPATIVEL: checkpoint R%lu, comando R%d. Troque experiment_id para novo experimento.\n",
            (unsigned long)h.completed_round, round);
        return -1;
    }
    File f=LittleFS.open(selected,"r");
    if(!f || !f.seek(sizeof(Header),SeekSet) || !m.loadWeights(f)) {
        if(f) f.close();
        Serial.println("[CKPT] ERRO na restauracao; nao executar treino com estado parcial.");
        return -1;
    }
    f.close();
    if(restored_hash) *restored_hash=h.checksum;
    Serial.printf("[CKPT] RESTAURADO experimento=%s rodada=%lu checksum=%08lX\n",
       experimentId(),(unsigned long)h.completed_round,(unsigned long)h.checksum);
    return 1;
}
static bool save(Model& m, int round) {
    if(!enabled()) return false; // Identidade obrigatoria para salvar.
    if(round<=0) return false;
    if(LittleFS.exists(TMP)) LittleFS.remove(TMP);
    File f=LittleFS.open(TMP,"w");
    if(!f) { Serial.println("[CKPT] Falha ao criar temporario."); return false; }
    Header h={MAGIC,VERSION,(uint32_t)CLIENT_ID,(uint32_t)MODEL_ARCH,
              experimentHash(),(uint32_t)round,modelBytes(),0};
    bool ok=f.write((uint8_t*)&h,sizeof(h))==sizeof(h) && m.saveWeights(f);
    f.flush();
    ok=ok && f.size()==sizeof(h)+(size_t)h.payload_bytes;
    f.close();
    if(!ok) { LittleFS.remove(TMP); Serial.println("[CKPT] Serializacao incompleta."); return false; }
    // Read full payload to calculate checksum; update header, then revalidate.
    File rd=LittleFS.open(TMP,"r");
    if(!rd || !rd.seek(sizeof(h),SeekSet) || !checksum(rd,h.payload_bytes,h.checksum)) {
        if(rd) rd.close(); LittleFS.remove(TMP); return false;
    }
    rd.close();
    File rw=LittleFS.open(TMP,"r+");
    if(!rw || !rw.seek(0,SeekSet) || rw.write((uint8_t*)&h,sizeof(h))!=sizeof(h)) {
        if(rw) rw.close(); LittleFS.remove(TMP); return false;
    }
    rw.flush(); rw.close();
    // verify uses the named slot; temporarily verify TMP before replacing anything.
    Header check={};
    if(!verify(TMP,check) || check.completed_round!=(uint32_t)round) {
        LittleFS.remove(TMP); Serial.println("[CKPT] Falha na verificacao."); return false;
    }
    const char* target=(round&1)?SLOT_A:SLOT_B;
    // Alternate slots: the other slot remains intact while publishing this one.
    if(LittleFS.exists(target) && !LittleFS.remove(target)) {
        LittleFS.remove(TMP); return false;
    }
    if(!LittleFS.rename(TMP,target)) {
        Serial.println("[CKPT] Falha na publicacao do checkpoint."); return false;
    }
    Serial.printf("[CKPT] SALVO rodada=%d bytes=%lu checksum=%08lX\n",round,
      (unsigned long)h.payload_bytes,(unsigned long)h.checksum);
    return true;
}
}
