_this_script=`readlink -f $0`
_this_dir=`dirname $_this_script`

python_env="/usr/prog/sb/mw/python/conda/sb_aegis/bin/python"
qsub_flags="-pe smp 1 -l gpu_card=1 -l gpu_arch=tesla_v100* -l h_rt=120000 -l m_mem_free=64G -cwd -j y -b y "


qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source iedb --features seq_only

qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source nod --features seq_only

qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source iedb_nod --features seq_only

qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source iedb --features seq_mhc

qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source nod --features seq_mhc

qsub $qsub_flags $python_env $_this_dir/../src/mhciipresentation/transformer.py --data_source iedb_nod --features seq_mhc

