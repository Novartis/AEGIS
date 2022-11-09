_this_script=`readlink -f $0`
_this_dir=`dirname $_this_script`
python_env="/usr/prog/sb/mw/python/conda/sb-aegis/bin/python"

$python_env ${_this_dir}/../src/mhciipresentation/evaluation/runAEGIS.py $@


