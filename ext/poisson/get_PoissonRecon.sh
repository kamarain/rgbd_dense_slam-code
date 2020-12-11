#!/bin/bash

if [ ! -f PoissonRecon.zip ];
then
	wget http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.71/PoissonRecon.zip
	unzip PoissonRecon.zip
fi
