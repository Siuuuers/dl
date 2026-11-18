#!/usr/bin/env python3
"""Losslessly compact review PDFs while verifying every embedded image.

8-bit DeviceRGB images are encoded with PNG row prediction and Flate. All image
pixel bytes, non-image decoded streams, page geometry, links, and bookmarks must
match after reopening the output. This does not resize or JPEG-encode images.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import struct
import sys

import fitz
from PIL import Image


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def png_idat(rgb, width, height):
    buffer = BytesIO()
    Image.frombytes('RGB', (width, height), rgb).save(buffer, format='PNG', compress_level=9)
    data = buffer.getvalue()
    require(data[:8] == b'\x89PNG\r\n\x1a\n', 'PNG encoder signature mismatch')
    offset, chunks = 8, []
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset+4])[0]
        kind = data[offset+4:offset+8]
        payload = data[offset+8:offset+8+length]
        if kind == b'IHDR':
            require(struct.unpack('>IIBBBBB', payload) == (width, height, 8, 2, 0, 0, 0),
                    'PNG encoder changed image dimensions, bit depth, color type or interlacing')
        elif kind == b'IDAT':
            chunks.append(payload)
        offset += length + 12
    require(chunks and offset == len(data), 'PNG chunk stream is invalid')
    return b''.join(chunks)


def image_receipt(document, xref):
    pixmap = fitz.Pixmap(document, xref)
    return {
        'xref': xref, 'width': pixmap.width, 'height': pixmap.height,
        'channels': pixmap.n, 'alpha': pixmap.alpha,
        'decoded_stream_sha256': digest(document.xref_stream(xref)),
        'decoded_pixels_sha256': digest(pixmap.samples),
    }


def page_identity(document):
    return [{'mediabox': tuple(page.mediabox), 'cropbox': tuple(page.cropbox),
             'rotation': page.rotation, 'links': page.get_links(),
             'contents': page.get_contents()} for page in document]


def reusable_image_streams(path):
    """Reuse previously verified predictor streams only by exact decoded identity."""
    result = {}
    document = fitz.open(path)
    for xref in range(1, document.xref_length()):
        if (document.xref_get_key(xref, 'Subtype') != ('name', '/Image')
                or document.xref_get_key(xref, 'ColorSpace') != ('name', '/DeviceRGB')
                or document.xref_get_key(xref, 'BitsPerComponent') != ('int', '8')
                or document.xref_get_key(xref, 'Filter') != ('name', '/FlateDecode')):
            continue
        width = int(document.xref_get_key(xref, 'Width')[1])
        height = int(document.xref_get_key(xref, 'Height')[1])
        parameters = dict((name, int(value)) for name, value in re.findall(
            r'/(Predictor|Colors|BitsPerComponent|Columns)\s+(\d+)',
            document.xref_get_key(xref, 'DecodeParms')[1]))
        if parameters != {'Predictor': 15, 'Colors': 3, 'BitsPerComponent': 8, 'Columns': width}:
            continue
        rgb = document.xref_stream(xref)
        require(len(rgb) == width * height * 3, f'Unexpected reusable RGB byte count: {xref}')
        result[(width, height, digest(rgb))] = document.xref_stream_raw(xref)
    document.close()
    return result


def optimize(source, output, audit_output, reuse_images_from=None):
    source, output, audit_output = map(lambda p: Path(p).resolve(), (source, output, audit_output))
    require(source.is_file() and source != output, 'Input must exist and output must be a different file')
    require(not output.exists() and not audit_output.exists(), 'Output and audit files must be new')
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    source_hash, source_size = file_digest(source), source.stat().st_size
    reusable = reusable_image_streams(reuse_images_from) if reuse_images_from else {}
    document = fitz.open(source)
    require(not document.is_encrypted, 'Encrypted input is not supported')
    image_refs = [xref for xref in range(1, document.xref_length())
                  if document.xref_get_key(xref, 'Subtype') == ('name', '/Image')]
    before_images = {xref: image_receipt(document, xref) for xref in image_refs}
    other_streams = {xref: digest(document.xref_stream(xref))
                     for xref in range(1, document.xref_length())
                     if xref not in before_images and document.xref_is_stream(xref)}
    pages_before = page_identity(document)
    bookmarks_before = document.get_toc(simple=False)
    rewritten, kept, reused = [], [], []
    for index, xref in enumerate(image_refs, 1):
        color = document.xref_get_key(xref, 'ColorSpace')
        bits = document.xref_get_key(xref, 'BitsPerComponent')
        if color != ('name', '/DeviceRGB') or bits != ('int', '8'):
            kept.append(xref)
            continue
        width = int(document.xref_get_key(xref, 'Width')[1])
        height = int(document.xref_get_key(xref, 'Height')[1])
        rgb = document.xref_stream(xref)
        require(len(rgb) == width * height * 3, f'Unexpected RGB sample byte count: image {xref}')
        key = (width, height, digest(rgb))
        compressed = reusable.get(key)
        if compressed is None:
            compressed = png_idat(rgb, width, height)
        else:
            reused.append(xref)
        document.update_stream(xref, compressed, compress=False)
        document.xref_set_key(xref, 'Filter', '/FlateDecode')
        document.xref_set_key(xref, 'DecodeParms',
                              f'<< /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {width} >>')
        require(image_receipt(document, xref) == before_images[xref],
                f'In-memory decoded image mismatch: {xref}')
        rewritten.append(xref)
        if index % 40 == 0:
            print(f'Losslessly rewrote {index}/{len(image_refs)} images ({len(reused)} reused)',
                  file=sys.stderr, flush=True)
    document.save(output, garbage=0, clean=False, deflate=False, no_new_id=True)
    document.close()

    after = fitz.open(output)
    after_image_refs = [xref for xref in range(1, after.xref_length())
                        if after.xref_get_key(xref, 'Subtype') == ('name', '/Image')]
    require(after_image_refs == image_refs, 'Embedded image identities changed')
    for xref in image_refs:
        require(image_receipt(after, xref) == before_images[xref],
                f'Saved decoded image mismatch: {xref}')
    for xref, expected in other_streams.items():
        require(after.xref_is_stream(xref) and digest(after.xref_stream(xref)) == expected,
                f'Non-image decoded stream changed: {xref}')
    require(page_identity(after) == pages_before, 'Page geometry, vector contents or links changed')
    require(after.get_toc(simple=False) == bookmarks_before, 'Bookmarks changed')
    require(file_digest(source) == source_hash, 'Original PDF changed during optimization')
    audit = {
        'schema_version': 1, 'status': 'verified_lossless',
        'source_pdf_sha256': source_hash, 'source_pdf_bytes': source_size,
        'output_pdf_sha256': file_digest(output), 'output_pdf_bytes': output.stat().st_size,
        'reduction_bytes': source_size - output.stat().st_size,
        'page_count': len(after), 'image_count': len(image_refs),
        'rgb_images_reencoded': len(rewritten), 'other_images_preserved': len(kept),
        'rgb_image_streams_reused': len(reused),
        'newly_encoded_rgb_images': len(rewritten) - len(reused),
        'reuse_source_pdf_sha256': file_digest(reuse_images_from) if reuse_images_from else None,
        'method': '8-bit DeviceRGB PNG row prediction (PDF Predictor 15) and FlateDecode',
        'all_decoded_image_streams_identical': True, 'all_decoded_image_pixels_identical': True,
        'non_image_decoded_streams_identical': True,
        'page_geometry_and_vector_contents_identical': True,
        'links_and_bookmarks_identical': True, 'source_pdf_unchanged': True,
        'no_jpeg_encoding': True, 'no_image_resizing': True,
        'optimizer_sha256': file_digest(__file__),
        'images': list(before_images.values()),
    }
    after.close()
    audit_output.write_text(json.dumps(audit, indent=2) + '\n')
    return {key: value for key, value in audit.items() if key != 'images'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--audit-output', required=True, type=Path)
    parser.add_argument('--reuse-images-from', type=Path,
                        help='optional prior optimized PDF; reuse only exact decoded RGB matches')
    args = parser.parse_args()
    print(json.dumps(optimize(args.input, args.output, args.audit_output, args.reuse_images_from), indent=2))


if __name__ == '__main__':
    main()
