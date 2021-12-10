#!/bin/sh
if [ -d venv ]
then
    . ./venv/bin/activate
else
    python3 -m venv venv
    . ./venv/bin/activate
    pip install -r requirements.txt
fi


# Run tasks
python task_1.py 2>/dev/null
python task_2.py 2>/dev/null
python task_3.py 2>/dev/null
