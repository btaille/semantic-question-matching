#!/bin/bash

for hid in 256 1024
do
	for train in successive joint
	do
		for restrict in 1000 5000 10000 25000 0
		do
			python training.py -m hybrid -res $restrict -tr $train -bs 16 -hid $hid
		done
	done
done

