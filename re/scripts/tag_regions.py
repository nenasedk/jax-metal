"""Tag every function in the jax-metal plugin as upstream (by component) or Apple code.

Input is the demangled symbol table produced by:

    nm -n DYLIB | awk ... > syms.tsv ; c++filt ... > syms_dem.tsv   (address, type, name)

Function sizes are the gap to the next text symbol, so they are approximate.

    python3 -I re/scripts/tag_regions.py re/work/syms_dem.tsv re/findings
"""

import collections
import os
import re
import sys

TEXT_END = 0x4CBA1BC  # end of __TEXT,__text (start of __stubs)

# Apple-authored code: own namespaces, Objective-C wrapper, C entry points.
APPLE = re.compile(
    r"^(mlir::mps::|xla::mps::|metal_plugin::|pjrt::metal_plugin::|MetalStream::|SP_Event_st::"
    r"|SP_Stream|-\[PjrtMetalstreamWrapper|_compileMlirHLOToMPS|compileMlirHLOToMPS"
    r"|_registerHLOToMPSPass|_registerMPSDialects|_GetPjrtApi|runMPSGraphExecutable"
    r"|invocation function for block in runMPSGraphExecutable)")
# Upstream templates instantiated over Apple types still count as Apple-driven code.
APPLE_TYPES = re.compile(r"(mlir|xla)::mps::|metal_plugin::|MetalStream|SP_Event_st")

UPSTREAM = [
    ("stablehlo", re.compile(r"^(mlir::(stablehlo|vhlo|chlo)::)")),
    ("xla/mlir-hlo", re.compile(r"^(mlir::(mhlo|lmhlo|lmhlo_gpu|gml_st|thlo)::)")),
    ("mlir", re.compile(r"^(mlir::|void mlir::|.* mlir::)")),
    ("llvm", re.compile(r"^(llvm::|void llvm::|.* llvm::|_LLVM|LLVM)")),
    ("xla", re.compile(r"^(xla::|stream_executor::|pjrt::|tsl::|tensorflow::|.* xla::)")),
    ("abseil", re.compile(r"^(absl::|.* absl::)")),
    ("protobuf", re.compile(r"^(google::protobuf::|google::|.* google::protobuf::)")),
    ("eigen", re.compile(r"^(Eigen::|.* Eigen::)")),
    ("libc++", re.compile(r"^(std::|void std::|.* std::)")),
    ("mlir-runtime", re.compile(r"^(_mlir|__mlir|_memref|_ciface)")),
]


def classify(name):
    if APPLE.search(name):
        return "apple"
    if APPLE_TYPES.search(name):
        return "apple-instantiation"
    for component, pattern in UPSTREAM:
        if pattern.search(name):
            return "upstream:" + component
    return "unattributed"


def main(symbols_path, out_dir):
    rows = []
    with open(symbols_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3 and parts[1] in ("t", "T"):
                rows.append((int(parts[0], 16), parts[2]))
    rows.sort()
    sized = []
    for index, (address, name) in enumerate(rows):
        if address >= TEXT_END:
            continue
        end = rows[index + 1][0] if index + 1 < len(rows) else TEXT_END
        sized.append((address, min(end, TEXT_END) - address, name, classify(name)))

    # File-local helpers ("(anonymous namespace)::X", outlined code) carry no namespace.
    # The linker keeps each object file contiguous, so inherit the tag of the nearest
    # attributed neighbours when both sides agree.
    tags = [tag for _, _, _, tag in sized]
    previous = [None] * len(tags)
    following = [None] * len(tags)
    last = None
    for index, tag in enumerate(tags):
        previous[index] = last
        if tag != "unattributed":
            last = tag
    last = None
    for index in range(len(tags) - 1, -1, -1):
        following[index] = last
        if tags[index] != "unattributed":
            last = tags[index]
    for index, (address, size, name, tag) in enumerate(sized):
        if tag == "unattributed" and previous[index] and previous[index] == following[index]:
            sized[index] = (address, size, name, previous[index] + " (by locality)")

    totals = collections.Counter()
    counts = collections.Counter()
    for _, size, _, tag in sized:
        totals[tag] += size
        counts[tag] += 1
    all_bytes = sum(totals.values())

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "apple-functions.tsv"), "w") as out:
        out.write("address\tsize\ttag\tname\n")
        for address, size, name, tag in sized:
            if tag.startswith("apple"):
                out.write(f"0x{address:x}\t{size}\t{tag}\t{name}\n")

    # Collapse consecutive functions with the same top-level tag into address ranges.
    ranges = []
    for address, size, _, tag in sized:
        top = "apple" if tag.startswith("apple") else "upstream"
        if ranges and ranges[-1][2] == top and ranges[-1][1] == address:
            ranges[-1][1] = address + size
            ranges[-1][3] += 1
        else:
            ranges.append([address, address + size, top, 1])
    with open(os.path.join(out_dir, "region-ranges.tsv"), "w") as out:
        out.write("start\tend\ttag\tfunctions\n")
        for start, end, tag, functions in ranges:
            out.write(f"0x{start:x}\t0x{end:x}\t{tag}\t{functions}\n")

    with open(os.path.join(out_dir, "region-summary.tsv"), "w") as out:
        out.write("tag\tfunctions\tbytes\tpercent_of_text\n")
        for tag, size in totals.most_common():
            out.write(f"{tag}\t{counts[tag]}\t{size}\t{100 * size / all_bytes:.2f}\n")
    print(open(os.path.join(out_dir, "region-summary.tsv")).read())
    apple_ranges = [r for r in ranges if r[2] == "apple"]
    print(f"{len(ranges)} ranges, {len(apple_ranges)} Apple ranges")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
