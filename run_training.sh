#!/bin/bash
echo ================ Training ================

# cleaning working directory
#rm step_count_log

read -p 'Enter the end step-number for training: ' max_stepnum
saved_stepnum=$(python2 ucv_utils.py)
echo Restoring model from $saved_stepnum steps.
i=$saved_stepnum
while [ !  $i -ge $max_stepnum  ]
do
	i=$(expr $i + 50000)
	echo --------------------------------------------------------
	echo Restarting training, running \until $i global steps.
	python2 ucv_control.py --mode='train' --steps=$i
	echo Training done, running evaluation.
	python2 ucv_control.py --mode='eval' --steps=0
	echo Plotting results.
	python2 trajectory_plotter --global_steps=$(expr $i / 1000)k
done


