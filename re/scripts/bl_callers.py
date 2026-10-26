"""Find the call sites of functions in an arm64 binary by scanning for BL instructions.

    python re/scripts/bl_callers.py BINARY TEXT_FILE_OFFSET TEXT_ADDRESS TEXT_SIZE TARGET...

TEXT_* describe the __text section (llvm-objdump -h; for Apple's plugin, offset and address
are both 0x6c00, size 0x4cb35bc). Each TARGET is a function address in hex, from llvm-nm.
Prints each target followed by the addresses of the BL instructions that call it. Calls
through pointers, tail calls (B) and jump tables are not found.
"""

import sys

import numpy as np


def main(path, offset, address, size, targets):
    with open(path, "rb") as handle:
        handle.seek(offset)
        words = np.frombuffer(handle.read(size), dtype="<u4")
    pcs = address + 4 * np.arange(len(words), dtype=np.int64)
    is_bl = (words >> 26) == 0b100101
    imm = (words & 0x3FFFFFF).astype(np.int64)
    imm = np.where(imm & 0x2000000, imm - 0x4000000, imm)  # sign-extend imm26
    destinations = pcs + 4 * imm
    for target in targets:
        hits = pcs[is_bl & (destinations == int(target, 16))]
        print(target, " ".join(hex(hit) for hit in hits))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2], 0), int(sys.argv[3], 0), int(sys.argv[4], 0), sys.argv[5:])
