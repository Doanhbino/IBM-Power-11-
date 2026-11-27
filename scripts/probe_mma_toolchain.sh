#!/usr/bin/env bash
# Read-only probe: does this host's compiler and CPU support the MMA paths we plan to use?
# Compiles tiny programs in a temp directory only. Changes nothing on the system.
set -uo pipefail
cc="${CXX:-g++}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "compiler: $($cc --version | head -1)"
echo "kernel:   $(uname -r)  machine: $(uname -m)"

cat > "$work/mma.cc" <<'SRC'
#include <altivec.h>
int main() { __vector_quad a; __vector unsigned char v{};
  __builtin_mma_xxsetaccz(&a); __builtin_mma_xvf32gerpp(&a, v, v);
  __vector float rows[4]; __builtin_mma_disassemble_acc(rows, &a); return 0; }
SRC
cat > "$work/hwcap.cc" <<'SRC'
#include <cstdio>
#include <sys/auxv.h>
#include <asm/cputable.h>
int main() { unsigned long h = getauxval(AT_HWCAP2);
  std::printf("AT_HWCAP2=0x%lx mma=%d arch_3_1=%d\n", h, !!(h & PPC_FEATURE2_MMA), !!(h & PPC_FEATURE2_ARCH_3_1)); return 0; }
SRC

try() {  # label, flags..., source
  local label="$1" src="$2"; shift 2
  if $cc -O2 "$@" "$src" -o "$work/a.out" 2>"$work/err.txt"; then echo "PASS  $label"; else
    echo "FAIL  $label: $(grep -m1 -i "error" "$work/err.txt")"; fi
}
try "MMA builtins compile with -mcpu=power10 -mmma" "$work/mma.cc" -mcpu=power10 -mmma
try "MMA builtins compile with -mcpu=power10 only"  "$work/mma.cc" -mcpu=power10
try "-mcpu=power11 accepted by this compiler"       "$work/mma.cc" -mcpu=power11 -mmma
if $cc -O2 "$work/hwcap.cc" -o "$work/hw" 2>"$work/err.txt"; then "$work/hw"; else
  echo "FAIL  hwcap probe did not compile: $(grep -m1 -i "error" "$work/err.txt")"; fi
