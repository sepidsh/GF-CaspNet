#!/usr/bin/env python
# -*- coding: utf-8 -*-



import cv2
import numpy as np

import os





from time import time

_tstart_stack = []


def tic():
	_tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
	print \
		fmt % (time() - _tstart_stack[0])

def toc_end(fmt="Elapsed: %s s"):
	print \
