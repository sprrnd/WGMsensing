#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:49:17 2024

@author: sabrinaperrenoud
"""

import picosdk

from picosdk.discover import find_all_units

scopes = find_all_units()

for scope in scopes:
    print(scope.info)
    scope.close()
