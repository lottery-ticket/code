#!/usr/bin/env python3

_db_config = {
    'user': 'root',
    'password': 'REDACTED',
    'host': '35.243.137.143',
    'database': 'lth_experiments',
    'raise_on_warnings': True,
}


import mysql.connector

conn = mysql.connector.connect(**_db_config)
cur = conn.cursor()
cur.execute('SELECT command FROM experiments WHERE status="completed" AND infrastructure="ibm"')
for (com,) in cur.fetchall():
    executable, network_type, prune_type, version, iter_no, idx, trial_no, _ = com.split(' ')

    if executable.endswith('lottery.sh'):
        typ_a = 'lottery'
        typ_b = 'prune'
    elif executable.endswith('finetune.sh'):
        typ_a = 'finetune'
        typ_b = 'finetune'
    elif executable.endswith('abbreviated.sh'):
        typ_a = 'lottery_early_40'
        typ_b = 'prune'
    elif executable.endswith('reinit.sh'):
        typ_a = 'reinit'
        typ_b = 'prune'
    else:
        raise ValueError(executable)

    if executable.startswith('vgg'):
        network = 'vgg_{}'.format(network_type)
    elif executable.startswith('resnet20'):
        network = 'resnet{}'.format(network_type)

    url = 's3://REDACTED-data{{region}}/execution_data/{network}/{prune_type}/{version}/oneshot_{typ_a}/{typ_b}_{idx}/trial_{trial_no}/iter_{iter_no}'.format(
        network=network,
        prune_type=prune_type,
        version=version,
        trial_no=trial_no,
        iter_no=iter_no,
        typ_a=typ_a,
        typ_b=typ_b,
        idx=idx,
    )

    print(url.format(region=''))
    print(url.format(region='-eu-gb'))
