#include "include/FedKDWire.h"
#include <cassert>
#include <cstdio>
int main(){uint8_t h[16];FedKDWire::build(h,3,"fedkd-bloco6-test-01"); assert(FedKDWire::valid(h,16,3,"fedkd-bloco6-test-01"));assert(!FedKDWire::valid(h,16,2,"fedkd-bloco6-test-01"));assert(!FedKDWire::valid(h,16,3,"other"));assert(FedKDWire::hashExperiment("a")==UINT64_C(0xaf63dc4c8601ec8c)); puts("PASS C++ wire");}
