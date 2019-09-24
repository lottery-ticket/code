#!/usr/bin/env python3

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+')
parser.add_argument('--region')
parser.add_argument('--endpoint')
parser.add_argument('--access-key-id')
parser.add_argument('--secret-access-key')
parser.add_argument('--use-https', default='1')
parser.add_argument('--verify-ssl', default='0')
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--collect', action='store_true', default=False)

args = parser.parse_args()

os.environ['AWS_REGION'] = args.region
os.environ['S3_ENDPOINT'] = args.endpoint
os.environ['AWS_ACCESS_KEY_ID'] = args.access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = args.secret_access_key
os.environ['S3_USE_HTTPS'] = args.use_https
os.environ['S3_VERIFY_SSL'] = args.verify_ssl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing
p = multiprocessing.Pool()

import utils
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

if args.collect:
    print(args.dirs, file=sys.stderr)
    print('\n'.join(utils.collect_trials(args.dirs)))
else:
    iters = p.starmap(
        utils.iter_datum_of_iter_dir,
        [((d, args.cache), False) for d in args.dirs]
    )
    print('\n'.join(map('{0[1]} {0[0].density_ratio} {0[0].test_acc}'.format, [(res, expt) for (res, expt) in zip(iters, args.dirs) if res is not None])))
