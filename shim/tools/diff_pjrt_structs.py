"""Compare PJRT C API structs between the plugin's header (0.47) and the framework's (0.116).

Reports, for every struct present in both headers, whether the newer one only appends
fields (safe to pass through to the old plugin) or changes existing fields (needs a
translation in the shim). Also lists PJRT_Api entries the plugin lacks.

    python3 -I shim/tools/diff_pjrt_structs.py OLD_HEADER NEW_HEADER
"""

import re
import sys


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def braced_body(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    raise ValueError("unbalanced braces")


def structs(text):
    text = strip_comments(text)
    found = {}
    for match in re.finditer(r"struct (\w+) \{", text):
        body = braced_body(text, match.end() - 1)
        # Flatten nested unions/structs so their members count as fields.
        body = re.sub(r"(union|struct)\s*\{", "", body).replace("}", "")
        fields = [re.sub(r"\s+", " ", f).strip() for f in body.split(";")]
        found[match.group(1)] = [f for f in fields if f and not f.startswith("#")]
    return found


def api_fields(text):
    body = strip_comments(text)
    end = body.index("} PJRT_Api;")
    body = re.sub(r"\s+", "", body[body.rindex("{", 0, body.rindex("_PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy)", 0, end)):end])
    return re.findall(r"_PJRT_API_STRUCT_FIELD\((\w+)\)", body)


def main(old_path, new_path):
    old_text, new_text = open(old_path).read(), open(new_path).read()
    old, new = structs(old_text), structs(new_text)
    appended, changed, same = [], [], []
    for name in sorted(set(old) & set(new)):
        before, after = old[name], new[name]
        if before == after:
            same.append(name)
        elif after[:len(before)] == before:
            appended.append((name, after[len(before):]))
        else:
            changed.append((name, before, after))

    print(f"# structs in both headers: {len(set(old) & set(new))}; "
          f"unchanged {len(same)}, append-only {len(appended)}, changed {len(changed)}\n")
    print("## Changed in place (needs review)\n")
    for name, before, after in changed:
        print(f"### {name}")
        for index in range(max(len(before), len(after))):
            a = before[index] if index < len(before) else ""
            b = after[index] if index < len(after) else ""
            if a != b:
                print(f"- [{index}] `{a}` -> `{b}`")
        print()
    print("## Append-only (new trailing fields the plugin ignores)\n")
    for name, extra in appended:
        print(f"- {name}: " + "; ".join(f"`{field}`" for field in extra))
    old_api, new_api = api_fields(old_text), api_fields(new_text)
    print(f"\n## PJRT_Api: plugin {len(old_api)} entries, framework {len(new_api)}; "
          f"prefix identical: {new_api[:len(old_api)] == old_api}\n")
    print("Missing from the plugin: " + ", ".join(new_api[len(old_api):]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
