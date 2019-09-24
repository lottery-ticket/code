#!/usr/bin/env python

import json
import os
import requests
import subprocess
import sys
import time

_DIRNAME = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

zone = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/zone', headers={'Metadata-Flavor': 'Google'}).text.split('/')[-1]

def tpu_status(name):
    tpus = json.loads(subprocess.check_output(['gcloud', 'compute', 'tpus', 'list', '--zone', zone, '--format', 'json']))

    return next(tpu for tpu in tpus if tpu['name'].split('/')[-1] == name)


def is_tpu_up(name):
    status = tpu_status(name)
    return status['state'] == 'READY' and status.get('health', 'UNKNOWN') == 'HEALTHY'

def assert_tpu_up(name):
    while not is_tpu_up(name):
        print('TPU {} is dead or unhealthy... Restarting...'.format(name))
        proc = subprocess.Popen(['gcloud', 'compute', 'tpus', 'stop', '--zone', zone, name])
        proc.wait()
        proc = subprocess.Popen(['gcloud', 'compute', 'tpus', 'start', '--zone', zone, name])
        if proc.wait() != 0:
            time.sleep(30)
    print('TPU {} running.'.format(name))

def main():
    tpu = sys.argv[1]
    program = sys.argv[2:]

    while True:
        assert_tpu_up(tpu)
        proc = subprocess.Popen(program)
        code = proc.wait()
        if is_tpu_up(tpu):
            return code

if __name__ == '__main__':
    sys.exit(main())
