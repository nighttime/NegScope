#!/bin/bash

BATCH=(10 30 50)
DROP_IN=(0.4 0.5 0.6)
DROP_REC=(0.1 0.2 0.3)
DROP_HID=(0.4 0.5 0.6)

for b in "${BATCH[@]}"; do
	for d_i in "${DROP_IN[@]}"; do
		for d_r in "${DROP_REC[@]}"; do
			for d_h in "${DROP_HID[@]}"; do
				echo "running" $h $b $d_i $d_r $d_h
			done
		done
	done
done
