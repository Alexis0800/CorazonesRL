"""Decode AXML binary layout files from Hearts APK."""
import struct
from pathlib import Path


def axml_to_text(filepath: str) -> str:
    with open(filepath, "rb") as f:
        data = f.read()

    magic, filesize = struct.unpack_from("<II", data, 0)
    if magic != 0x00080003:
        return f"Not AXML: 0x{magic:08x}"

    strings: list[str] = []
    pos = 8  # skip AXML header, start at first sub-chunk
    result: list[str] = []
    INDENT = "  "

    while pos < len(data) - 8:
        chunk_type, chunk_size = struct.unpack_from("<II", data, pos)

        if chunk_type == 0x001C0001:  # StringPool
            str_count = struct.unpack_from("<I", data, pos + 8)[0]
            style_count = struct.unpack_from("<I", data, pos + 12)[0]
            flags = struct.unpack_from("<I", data, pos + 16)[0]
            str_data_off = struct.unpack_from("<I", data, pos + 20)[0]
            utf8 = bool(flags & (1 << 8))

            offset_base = pos + 28 + (str_count + style_count) * 4
            for i in range(str_count):
                off = struct.unpack_from("<I", data, pos + 28 + i * 4)[0]
                abs_off = offset_base + off
                if utf8:
                    b1 = data[abs_off]
                    if abs_off + 1 >= len(data):
                        strings.append("")
                        continue
                    b2 = data[abs_off + 1]
                    if b1 & 0x80:
                        slen = ((b1 & 0x7F) << 8) | b2
                        sbytes = data[abs_off + 2: abs_off + 2 + slen]
                    else:
                        slen = b1
                        sbytes = data[abs_off + 1: abs_off + 1 + slen]
                    strings.append(sbytes.decode("utf-8", errors="replace"))
                else:
                    slen = struct.unpack_from("<H", data, abs_off)[0]
                    sbytes = data[abs_off + 2: abs_off + 2 + slen * 2]
                    strings.append(sbytes.decode(
                        "utf-16-le", errors="replace"))

        elif chunk_type == 0x00100100:  # XML tree
            p = pos + 8  # skip chunk header
            chunk_end = pos + chunk_size
            depth = 0

            while p < chunk_end - 4:
                node_type = struct.unpack_from("<I", data, p)[0]
                if node_type == 0x00100102:  # StartTag
                    _, _, line, _, ns_idx, name_idx = struct.unpack_from(
                        "<IHHIII", data, p
                    )
                    attr_count, _ = struct.unpack_from("<II", data, p + 24)
                    tag = (
                        strings[name_idx]
                        if name_idx < len(strings)
                        else f"?{name_idx}"
                    )
                    attrs: list[str] = []
                    for a in range(attr_count):
                        ao = p + 36 + a * 20
                        a_ns, a_name, a_val_s, a_flags, a_data = (
                            struct.unpack_from("<IIIiI", data, ao)
                        )
                        a_name_s = (
                            strings[a_name]
                            if a_name < len(strings)
                            else f"?{a_name}"
                        )
                        # Determine attribute value
                        if a_val_s != 0xFFFFFFFF and a_val_s < len(strings):
                            a_val = strings[a_val_s]
                        elif (a_flags >> 24) in (0x10, 0x11, 0x12):
                            a_val = str(a_data)
                        elif (a_flags >> 24) == 0x01:
                            a_val = f"@res/0x{a_data:08x}"
                        else:
                            a_val = f"0x{a_data:08x}"
                        attrs.append(f'{a_name_s}="{a_val}"')

                    indent = INDENT * depth
                    result.append(f"{indent}<{tag} line={line}")
                    for a in attrs:
                        result.append(f"{indent}{INDENT}{a}")
                    depth += 1
                    p += 36 + attr_count * 20

                elif node_type == 0x00100103:  # EndTag
                    depth -= 1
                    _, _, line, _, ns_idx, name_idx = struct.unpack_from(
                        "<IHHIII", data, p
                    )
                    tag = (
                        strings[name_idx]
                        if name_idx < len(strings)
                        else f"?{name_idx}"
                    )
                    indent = INDENT * depth
                    result.append(f"{indent}</{tag}>")
                    p += 24

                elif node_type == 0x00100104:  # Text node
                    _, _, line, _, ns_idx, name_idx = struct.unpack_from(
                        "<IHHIII", data, p
                    )
                    text = (
                        strings[name_idx]
                        if name_idx < len(strings)
                        else f"?{name_idx}"
                    )
                    indent = INDENT * depth
                    result.append(f'{indent}TEXT="{text}"')
                    p += 28

                elif node_type == 0x00100101:  # Namespace start
                    _, _, line, _, prefix_idx, uri_idx = struct.unpack_from(
                        "<IHHIII", data, p
                    )
                    prefix = strings[prefix_idx] if prefix_idx < len(
                        strings) else ""
                    uri = strings[uri_idx] if uri_idx < len(strings) else ""
                    indent = INDENT * depth
                    result.append(f'{indent}xmlns:{prefix}="{uri}"')
                    p += 24

                else:
                    break

        pos += chunk_size

    return "\n".join(result)


def main() -> None:
    layouts = Path("calibracion/Corazones/recursos/layouts")
    files = ["room.xml", "activity.xml",
             "rules_description.xml", "rules_settings.xml"]
    for fname in files:
        path = layouts / fname
        if not path.exists():
            print(f"===== {fname} ===== MISSING\n")
            continue
        print(f"===== {fname} =====")
        print(axml_to_text(str(path)))
        print()


if __name__ == "__main__":
    main()
