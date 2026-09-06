#include <nvtx3/nvToolsExt.h>
#include <stdint.h>

static nvtxDomainHandle_t domain() {
    static auto d = nvtxDomainCreateA("alpharat");
    return d;
}
extern "C" void alpharat_nvtx_push(const char* name, uint64_t id) {
    nvtxEventAttributes_t e = {};
    e.version = NVTX_VERSION;
    e.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.messageType = NVTX_MESSAGE_TYPE_ASCII;
    e.message.ascii = name;
    e.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    e.payload.ullValue = id;
    nvtxDomainRangePushEx(domain(), &e);
}
extern "C" void alpharat_nvtx_pop() { nvtxDomainRangePop(domain()); }
