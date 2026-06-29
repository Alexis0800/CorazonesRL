"""AXML decoder handling per-node chunks (actual Hearts APK format)."""
import struct
from pathlib import Path


def decode_string_pool(data: bytes, pos: int) -> tuple[list[str], int]:
    _, chunk_size = struct.unpack_from("<II", data, pos)
    str_count = struct.unpack_from("<I", data, pos + 8)[0]
    style_count = struct.unpack_from("<I", data, pos + 12)[0]
    flags = struct.unpack_from("<I", data, pos + 16)[0]
    utf8 = bool(flags & (1 << 8))

    strings = []
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
                prefix = 2
            else:
                slen = b1
                prefix = 1
            sbytes = data[abs_off + prefix: abs_off + prefix + slen]
            # MUTF-8: length includes null terminator; strip trailing null
            s = sbytes.decode("utf-8", errors="replace").rstrip('\x00')
            strings.append(s)
        else:
            slen = struct.unpack_from("<H", data, abs_off)[0]
            sbytes = data[abs_off + 2: abs_off + 2 + slen * 2]
            strings.append(sbytes.decode(
                "utf-16-le", errors="replace").rstrip("\0"))
    return strings, pos + chunk_size


def decode_xml_node(data: bytes, pos: int, strings: list[str]) -> tuple | None:
    """Parse one XML node chunk."""
    node_type, chunk_size = struct.unpack_from("<II", data, pos)

    # Safety: need at least 24 bytes (min header) from pos
    needed = 24
    if pos + needed > len(data):
        return None

    line = struct.unpack_from("<I", data, pos + 8)[0]
    ns_idx = struct.unpack_from("<i", data, pos + 16)[0]
    name_idx = struct.unpack_from("<i", data, pos + 20)[0]

    if node_type == 0x00100102:  # START_TAG (8 + 20 + 2 + 2 + attrs*20)
        if pos + 30 > len(data):
            return None
        attr_count, _ = struct.unpack_from("<HH", data, pos + 28)
        tag = strings[name_idx] if 0 <= name_idx < len(
            strings) else f"?{name_idx}"
        attrs = {}
        for a in range(attr_count):
            ao = pos + 30 + a * 20
            if ao + 20 > len(data):
                break
            ns_i, a_name, a_val_s, a_flags, a_data = struct.unpack_from(
                "<IIIiI", data, ao)
            aname = strings[a_name] if 0 <= a_name < len(
                strings) else f"?{a_name}"
            if a_val_s != -1 and a_val_s < len(strings):
                aval = strings[a_val_s]
            elif (a_flags >> 24) in (0x10, 0x11, 0x12):
                aval = str(a_data)
            elif (a_flags >> 24) == 0x01:
                aval = f"@res/0x{a_data:08x}"
            else:
                aval = f"0x{a_data:08x}"
            attrs[aname] = aval
        return ("start", tag, attrs, "")
    elif node_type == 0x00100103:  # END_TAG (24 bytes)
        tag = strings[name_idx] if 0 <= name_idx < len(
            strings) else f"?{name_idx}"
        return ("end", tag, {}, "")
    elif node_type == 0x00100104:  # TEXT
        text = strings[name_idx] if 0 <= name_idx < len(strings) else ""
        return ("text", "", {}, text)
    elif node_type == 0x00100101:  # NAMESPACE (24 bytes)
        prefix = strings[name_idx] if 0 <= name_idx < len(strings) else ""
        uri = strings[ns_idx] if 0 <= ns_idx < len(strings) else ""
        return ("ns", prefix, {"uri": uri}, "")
    return None


def axml_to_text(filepath: str) -> str:
    data = Path(filepath).read_bytes()
    strings = []

    # Pass 1: string pool
    pos = 8
    while pos < len(data) - 8:
        ctype = struct.unpack_from("<I", data, pos)[0]
        if ctype == 0x001C0001:
            strings, pos = decode_string_pool(data, pos)
            break
        pos += struct.unpack_from("<I", data, pos + 4)[0]

    # Pass 2: XML nodes
    events = []
    pos = 8
    while pos < len(data) - 8:
        ctype = struct.unpack_from("<I", data, pos)[0]
        csize = struct.unpack_from("<I", data, pos + 4)[0]
        if ctype in (0x00100102, 0x00100103, 0x00100104, 0x00100101):
            node = decode_xml_node(data, pos, strings)
            if node:
                events.append(node)
        pos += csize

    # Build text tree
    IND = "  "
    lines = []
    depth = 0
    open_tags = []
    prev_was_start = False

    for evt in events:
        typ = evt[0]
        if typ == "ns":
            lines.append(
                f'{IND * depth}xmlns:{evt[1]}="{evt[2].get("uri", "")}"')
        elif typ == "start":
            tag = evt[1]
            attrs = evt[2]
            indent = IND * depth
            attr_parts = []
            for k, v in attrs.items():
                vs = v.replace("match_parent", "MP").replace(
                    "wrap_content", "WC")
                attr_parts.append(f'{k}="{vs}"')
            # Important attrs: id, text, background, visibility, orientation
            key_attrs = [a for a in attr_parts if any(
                a.startswith(k) for k in ["id=", "text=", "orientation=",
                                          "background=", "visibility="])]
            other_attrs = [a for a in attr_parts if a not in key_attrs]
            display_attrs = key_attrs + other_attrs[:4]  # show first 4 non-key
            if len(other_attrs) > 4:
                display_attrs.append(f'...(+{len(other_attrs)-4})')

            lines.append(f"{indent}<{tag}>")
            for a in display_attrs:
                lines.append(f"{indent}{IND}{a}")
            open_tags.append(tag)
            depth += 1
            prev_was_start = True
        elif typ == "end":
            depth -= 1
            tag = evt[1]
            if open_tags and open_tags[-1] == tag:
                open_tags.pop()
            lines.append(f"{IND * depth}</{tag}>")
            prev_was_start = False
        elif typ == "text":
            txt = evt[3].strip()
            if txt:
                lines.append(f'{IND * depth}"{txt}"')

    return "\n".join(lines)


def main():
    layouts = Path("calibracion/Corazones/recursos/layouts")
    for fname in ["room.xml", "activity.xml", "rules_description.xml", "rules_settings.xml"]:
        path = layouts / fname
        if not path.exists():
            continue
        print(f"===== {fname} =====")
        print(axml_to_text(str(path)))
        print()


if __name__ == "__main__":
    main()
