#!/bin/bash
TESTCASES="c01.1"
make
./hw3-2 testcases/${TESTCASES} ${TESTCASES}.out
diff <(hw3-cat ${TESTCASES}.out) <(hw3-cat testcases/${TESTCASES}.out)