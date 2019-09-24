#!/usr/bin/env python

import struct
import sys

def bytestr(orig):
    norm = '{:x}'.format(int(orig, 0))
    if len(norm) % 2 == 1:
        norm = '0' + norm
    ints = []

    for i in range(len(norm)//2):
        ints.append(int('0x'+norm[i*2:i*2+2], 0))

    return struct.pack('>'+'B'*len(ints), *ints)


fname, offset = sys.argv[1].split(':')
offset = int(offset, 0)
b = bytestr(sys.argv[2])

with open(fname, 'r+b') as f:
    f.seek(offset)
    f.write(b)
