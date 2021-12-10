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
python task_1.py 
python task_2.py
python task_3.py
