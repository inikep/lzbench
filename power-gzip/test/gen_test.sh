#!/bin/bash

mode=$1
test=$2

cat <<EOF
#!/bin/bash
NX_GZIP_TYPE_SELECTOR=${mode} ./${test}
EOF
