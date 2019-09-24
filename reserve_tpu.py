#!/usr/bin/env python

import argparse
import json
import os
import re
import requests
import subprocess
import sys

zone = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/zone', headers={'Metadata-Flavor': 'Google'}).text.split('/')[-1]
name = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/name', headers={'Metadata-Flavor': 'Google'}).text

tpu_re = re.compile(r'^projects/(?P<project>.+?)/locations/(?P<zone>.+?)/nodes/(?P<name>.*?)-(?P<index>\d+)$')
tpu_dir = os.path.join(os.environ['HOME'], '.tpus')
os.makedirs(tpu_dir, exist_ok=True)

def is_free(idx):
    return not os.path.exists(os.path.join(tpu_dir, idx))

def reserve(pid):
    tpus = json.loads(subprocess.check_output([
        'gcloud', 'compute', 'tpus', 'list',
        '--zone', zone, '--format', 'json',
    ]))
    valid_tpu_matches = filter(lambda x: x is not None, (tpu_re.match(tpu['name']) for tpu in tpus))
    my_tpu_matches = (tpu for tpu in valid_tpu_matches if tpu.group('name') == name)
    free_tpu_indices = (tpu.group('index') for tpu in my_tpu_matches if is_free(tpu.group('index')))

    while True:
        try:
            idx = next(free_tpu_indices)
            with open(os.path.join(tpu_dir, idx), 'x') as f:
                f.write('{}\n'.format(pid))
            break
        except FileExistsError:
            continue

    print(idx)

def release(idx):
    os.unlink(os.path.join(tpu_dir, str(idx)))

def cleanup():
    for fname in os.listdir(tpu_dir):
        with open(os.path.join(tpu_dir, fname)) as f:
            pid = f.read().strip()
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            os.unlink(os.path.join(tpu_dir, fname))

def main():
    parser = argparse.ArgumentParser(description='Reserve TPUs locally to avoid conflicts')
    subp = parser.add_subparsers(dest='_subparser')
    reserve_parser = subp.add_parser('reserve', help='Reserve a TPU')
    reserve_parser.add_argument('pid', type=int, help='PID of reserving process (when this pid is dead, `cleanup` will release the reserved TPU')
    release_parser = subp.add_parser('release', help='Release a previously reserved TPU')
    release_parser.add_argument('idx', type=int, help='TPU index to release')
    cleanup_parser = subp.add_parser('cleanup', help='Cleanup dead TPU reservations')

    args = parser.parse_args()

    if args._subparser == 'reserve':
        reserve(args.pid)
    elif args._subparser == 'release':
        release(args.idx)
    elif args._subparser == 'cleanup':
        cleanup()
    else:
        raise ValueError('Unknown subparser {}'.format(args._subparser))

if __name__ == '__main__':
    main()
