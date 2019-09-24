#!/usr/bin/env python3

import base64
import subprocess
import json

s = set()

while True:
    m = json.loads(subprocess.check_output('gcloud --project carbingroup pubsub subscriptions pull REDACTED --format json --limit 250', shell=True))
    bad = []
    bad_m = []
    for mess in m:
        z = base64.b64decode(mess['message']['data'])
        s.add(z)
        bad_m.append(z.decode('utf-8'))
        bad.append(mess['ackId'])

    print(bad)
    if bad:
        subprocess.check_call('gcloud --project carbingroup pubsub subscriptions ack REDACTED --ack-ids={}'.format(','.join(bad)), shell=True)
        print('\n'.join(bad_m))
