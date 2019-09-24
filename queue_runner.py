#!/usr/bin/env python

from __future__ import print_function

import base64
import json
import os
import requests
import subprocess
import time
import traceback
import tensorflow as tf
import threading
import queue

node_id = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/name', headers={'Metadata-Flavor': 'Google'}).text
equals = ('='*100)+'\n'

os.chdir('/home/lth/lth')

pid = os.getpid()

os.environ['QUEUE_PID_NAME'] = '{}-{}'.format(node_id, pid)
stdout_fname = 'gs://REDACTED/queue_logs/{}-{}.stdout'.format(node_id, pid)
stderr_fname = 'gs://REDACTED/queue_logs/{}-{}.stderr'.format(node_id, pid)

with tf.gfile.Open(stdout_fname, 'w') as stdout, tf.gfile.Open(stderr_fname, 'w') as stderr:
    print('{equals}\nStarting queue on {node_id}-{pid}\n{equals}'.format(equals=equals, pid=pid, node_id=node_id), file=stdout)
    print('{equals}\nStarting queue on {node_id}-{pid}\n{equals}'.format(equals=equals, pid=pid, node_id=node_id), file=stderr)


def check_unique_processor(message_id):
    # solve consensus lol

    identifier = '{}-{}'.format(node_id, pid)
    message_fname = 'gs://REDACTED/pubsub_messages/{}'.format(message_id)

    if tf.gfile.Exists(message_fname):
        return False

    with tf.gfile.Open(message_fname, 'w') as f:
        f.write(identifier)

    time.sleep(10)

    with tf.gfile.Open(message_fname) as f:
        owner = f.read().strip()

    return owner == identifier

last_write_time = None

def piper(command, from_, to_fname):
    global last_write_time

    q = queue.Queue()
    def reader():
        global last_write_time
        while True:
            last_write_time = time.time()
            line = from_.readline()
            if line:
                q.put(line)
            if not line.endswith(b'\n'):
                q.put(None)
                return

    threading.Thread(target=reader).start()

    with tf.gfile.Open(to_fname, 'a') as to:
        print(equals+command+'\n'+equals, file=to)
        while True:
            line = q.get()
            if line is None:
                return
            to.write(line)
            if q.qsize() == 0:
                to.flush()


def ping_slack(message):
    while True:
        try:
            subprocess.check_call(['/home/lth/lth/REDACTED/cloud.py', 'slack', message])
            break
        except subprocess.CalledProcessError:
            pass

while True:
    out = subprocess.check_output(['gcloud', 'pubsub', 'subscriptions', 'pull', '--auto-ack', 'REDACTED', '--format', 'json'], universal_newlines=True).strip()
    out = json.loads(out)
    if len(out) == 0:
        time.sleep(5)
        continue

    last_write_time = time.time()

    command = base64.b64decode(out[0]['message']['data']).decode('utf-8')

    if not check_unique_processor(out[0]['message']['messageId']):
        ping_slack('`{node_id}-{pid}` skipping `{command}`'.format(
            node_id=node_id,
            pid=pid,
            command=command,
        ))
        continue


    ping_slack('`{node_id}-{pid}` starting `{command}`'.format(
        node_id=node_id,
        pid=pid,
        command=command,
    ))
    try:
        p = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        threading.Thread(target=piper, args=(command, p.stdout, stdout_fname)).start()
        threading.Thread(target=piper, args=(command, p.stderr, stderr_fname)).start()

        while True:
            try:
                code = p.wait(60)
                break
            except subprocess.TimeoutExpired:
                # if we must wait for over 30 minutes
                if time.time() - last_write_time > 30 * 60:
                    ping_slack('`{node_id}-{pid}` killed `{command}` due to timeout.\n'
                               'Re-queueing...'.format(
                                   node_id=node_id,
                                   pid=pid,
                                   command=command,
                    ))
                    # Terminate PID and child processes
                    os.kill(-p.pid, 15)

        ping_slack('`{node_id}-{pid}` finished `{command}`: {code}'.format(
            node_id=node_id,
            pid=pid,
            command=command,
            code=':white_check_mark:' if code == 0 else ':x: ({})'.format(code),
        ))

        if code not in (0, -2):
            subprocess.check_call([
                'gcloud', 'pubsub', 'topics', 'publish', 'REDACTED', '--message',
                command
            ])
    except:
        ping_slack('`{node_id}-{pid}` errored on `{command}`! ```{error}```'.format(
            node_id=node_id,
            pid=pid,
            command=command,
            error=traceback.format_exc()
        ))
    # wait 10s before pulling again
    time.sleep(10)
