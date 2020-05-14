#!/usr/bin/env python

"""
  lk_checkers.py
"""

from numba import njit

# --------------------------------------------------------------------------------
# Wrapper

@njit(cache=True)
def check_move(csh, n_nodes):
    if len(csh) == 4:
        return check2(csh)
    elif len(csh) == 6:
        return check3(csh)
    elif len(csh) == 8:
        return check4(csh)
    elif len(csh) == 10:
        return check5(csh)
    else:
        raise Exception('check_move: `cs` is unsupported length')

# --------------------------------------------------------------------------------
# Move Checkers
# - These are automatically generated functions that determine whether a move `cs` yields a connected route
# - A more human-friendly way to implement these checkers would be to check whether `cs` is in a set of possible moves
#   However, I was getting performance penalties implementing that in numba.
# - Code to generate these functions can be found in `tools/enumerate_lk.py`

@njit(cache=True)
def check2(cs):
  if cs[0] <= cs[1]:
    if cs[1] <= cs[3]:
      if cs[3] <= cs[2]:
         return True
  
  return False

@njit(cache=True)
def check3(cs):
  if cs[0] <= cs[1]:
    if cs[1] <= cs[3]:
      if cs[3] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[4]:
             return True
    if cs[1] <= cs[4]:
      if cs[4] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[3]:
             return True
      if cs[4] <= cs[5]:
        if cs[5] <= cs[2]:
          if cs[2] <= cs[3]:
             return True
        if cs[5] <= cs[3]:
          if cs[3] <= cs[2]:
             return True
    if cs[1] <= cs[5]:
      if cs[5] <= cs[4]:
        if cs[4] <= cs[2]:
          if cs[2] <= cs[3]:
             return True
  
  return False

@njit(cache=True)
def check4(cs):
  if cs[0] <= cs[1]:
    if cs[1] <= cs[3]:
      if cs[3] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[6]:
                 return True
        if cs[2] <= cs[6]:
          if cs[6] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[5]:
                 return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
            if cs[7] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
        if cs[2] <= cs[7]:
          if cs[7] <= cs[6]:
            if cs[6] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
    if cs[1] <= cs[4]:
      if cs[4] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                 return True
      if cs[4] <= cs[5]:
        if cs[5] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                 return True
        if cs[5] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[7]:
              if cs[7] <= cs[6]:
                 return True
        if cs[5] <= cs[7]:
          if cs[7] <= cs[2]:
            if cs[2] <= cs[6]:
              if cs[6] <= cs[3]:
                 return True
          if cs[7] <= cs[3]:
            if cs[3] <= cs[6]:
              if cs[6] <= cs[2]:
                 return True
          if cs[7] <= cs[6]:
            if cs[6] <= cs[2]:
              if cs[2] <= cs[3]:
                 return True
            if cs[6] <= cs[3]:
              if cs[3] <= cs[2]:
                 return True
    if cs[1] <= cs[5]:
      if cs[5] <= cs[4]:
        if cs[4] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                 return True
        if cs[4] <= cs[6]:
          if cs[6] <= cs[2]:
            if cs[2] <= cs[7]:
              if cs[7] <= cs[3]:
                 return True
          if cs[6] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[2]:
                 return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[2]:
              if cs[2] <= cs[3]:
                 return True
            if cs[7] <= cs[3]:
              if cs[3] <= cs[2]:
                 return True
        if cs[4] <= cs[7]:
          if cs[7] <= cs[3]:
            if cs[3] <= cs[6]:
              if cs[6] <= cs[2]:
                 return True
          if cs[7] <= cs[6]:
            if cs[6] <= cs[3]:
              if cs[3] <= cs[2]:
                 return True
    if cs[1] <= cs[6]:
      if cs[6] <= cs[2]:
        if cs[2] <= cs[7]:
          if cs[7] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
      if cs[6] <= cs[3]:
        if cs[3] <= cs[7]:
          if cs[7] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
      if cs[6] <= cs[4]:
        if cs[4] <= cs[7]:
          if cs[7] <= cs[5]:
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                 return True
      if cs[6] <= cs[7]:
        if cs[7] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
        if cs[7] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
        if cs[7] <= cs[4]:
          if cs[4] <= cs[5]:
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                 return True
        if cs[7] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                 return True
    if cs[1] <= cs[7]:
      if cs[7] <= cs[2]:
        if cs[2] <= cs[6]:
          if cs[6] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
      if cs[7] <= cs[3]:
        if cs[3] <= cs[6]:
          if cs[6] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
      if cs[7] <= cs[6]:
        if cs[6] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
        if cs[6] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                 return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                 return True
        if cs[6] <= cs[4]:
          if cs[4] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                 return True
          if cs[4] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                 return True
  
  return False


@njit(cache=True)
def check5(cs):
  if cs[0] <= cs[1]:
    if cs[1] <= cs[3]:
      if cs[3] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[4] <= cs[8]:
              if cs[8] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[7]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[9] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[4] <= cs[9]:
              if cs[9] <= cs[8]:
                if cs[8] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
        if cs[2] <= cs[6]:
          if cs[6] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[7] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[7] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
        if cs[2] <= cs[7]:
          if cs[7] <= cs[6]:
            if cs[6] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[6] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[6] <= cs[9]:
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
        if cs[2] <= cs[8]:
          if cs[8] <= cs[4]:
            if cs[4] <= cs[9]:
              if cs[9] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[8] <= cs[5]:
            if cs[5] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[8] <= cs[6]:
            if cs[6] <= cs[9]:
              if cs[9] <= cs[7]:
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[8] <= cs[9]:
            if cs[9] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[9] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[9] <= cs[6]:
              if cs[6] <= cs[7]:
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[9] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[2] <= cs[9]:
          if cs[9] <= cs[4]:
            if cs[4] <= cs[8]:
              if cs[8] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[9] <= cs[5]:
            if cs[5] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[9] <= cs[8]:
            if cs[8] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[8] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[8] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
    if cs[1] <= cs[4]:
      if cs[4] <= cs[2]:
        if cs[2] <= cs[5]:
          if cs[5] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[7]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[9] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[8]:
                if cs[8] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
      if cs[4] <= cs[5]:
        if cs[5] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[7]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[9] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[8]:
                if cs[8] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
        if cs[5] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[8]:
              if cs[8] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[7]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[9] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[9]:
              if cs[9] <= cs[8]:
                if cs[8] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
        if cs[5] <= cs[7]:
          if cs[7] <= cs[2]:
            if cs[2] <= cs[6]:
              if cs[6] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[7] <= cs[3]:
            if cs[3] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[7] <= cs[6]:
            if cs[6] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[6] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[6] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[8]:
                  if cs[8] <= cs[3]:
                     return True
              if cs[9] <= cs[3]:
                if cs[3] <= cs[8]:
                  if cs[8] <= cs[2]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[8] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[5] <= cs[8]:
          if cs[8] <= cs[2]:
            if cs[2] <= cs[9]:
              if cs[9] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[8] <= cs[3]:
            if cs[3] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[8] <= cs[6]:
            if cs[6] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[9] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[9] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[8] <= cs[9]:
            if cs[9] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[9] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[9] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[6] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[9] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[5] <= cs[9]:
          if cs[9] <= cs[2]:
            if cs[2] <= cs[8]:
              if cs[8] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[9] <= cs[3]:
            if cs[3] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[9] <= cs[8]:
            if cs[8] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[8] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[8] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[6] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
    if cs[1] <= cs[5]:
      if cs[5] <= cs[4]:
        if cs[4] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[6]:
                if cs[6] <= cs[9]:
                  if cs[9] <= cs[7]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[9] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[8]:
                if cs[8] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
        if cs[4] <= cs[6]:
          if cs[6] <= cs[2]:
            if cs[2] <= cs[7]:
              if cs[7] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[6] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[7] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[7] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[8]:
                  if cs[8] <= cs[3]:
                     return True
              if cs[9] <= cs[3]:
                if cs[3] <= cs[8]:
                  if cs[8] <= cs[2]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[8] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[4] <= cs[7]:
          if cs[7] <= cs[3]:
            if cs[3] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[7] <= cs[6]:
            if cs[6] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[6] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[3]:
                     return True
              if cs[8] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[2]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[9] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[6] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[8]:
                  if cs[8] <= cs[3]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[4] <= cs[8]:
          if cs[8] <= cs[2]:
            if cs[2] <= cs[9]:
              if cs[9] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[8] <= cs[3]:
            if cs[3] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[8] <= cs[6]:
            if cs[6] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[9] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
          if cs[8] <= cs[9]:
            if cs[9] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[9] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[9] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
            if cs[9] <= cs[7]:
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[4] <= cs[9]:
          if cs[9] <= cs[2]:
            if cs[2] <= cs[8]:
              if cs[8] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[9] <= cs[3]:
            if cs[3] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[9] <= cs[8]:
            if cs[8] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[8] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[8] <= cs[6]:
              if cs[6] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
    if cs[1] <= cs[6]:
      if cs[6] <= cs[2]:
        if cs[2] <= cs[7]:
          if cs[7] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[6] <= cs[3]:
        if cs[3] <= cs[7]:
          if cs[7] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[2] <= cs[9]:
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
      if cs[6] <= cs[4]:
        if cs[4] <= cs[7]:
          if cs[7] <= cs[5]:
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[5] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[3]:
                     return True
              if cs[8] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[2]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[9] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[5] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[8]:
                  if cs[8] <= cs[3]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
      if cs[6] <= cs[7]:
        if cs[7] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[7] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[2] <= cs[9]:
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
        if cs[7] <= cs[4]:
          if cs[4] <= cs[5]:
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[5] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[3]:
                     return True
              if cs[8] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[2]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[9] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[5] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[8]:
                  if cs[8] <= cs[3]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[7] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[4] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[3]:
                     return True
              if cs[8] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[2]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[9] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[4] <= cs[9]:
              if cs[9] <= cs[3]:
                if cs[3] <= cs[8]:
                  if cs[8] <= cs[2]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[7] <= cs[9]:
          if cs[9] <= cs[2]:
            if cs[2] <= cs[8]:
              if cs[8] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[9] <= cs[3]:
            if cs[3] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
          if cs[9] <= cs[4]:
            if cs[4] <= cs[8]:
              if cs[8] <= cs[5]:
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[9] <= cs[5]:
            if cs[5] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
          if cs[9] <= cs[8]:
            if cs[8] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[8] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
            if cs[8] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[8] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
    if cs[1] <= cs[7]:
      if cs[7] <= cs[2]:
        if cs[2] <= cs[6]:
          if cs[6] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[7] <= cs[3]:
        if cs[3] <= cs[6]:
          if cs[6] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
      if cs[7] <= cs[6]:
        if cs[6] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[3] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[5]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[4]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[9] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[6] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[2] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[8]:
                  if cs[8] <= cs[5]:
                     return True
              if cs[9] <= cs[5]:
                if cs[5] <= cs[8]:
                  if cs[8] <= cs[4]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[8] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
        if cs[6] <= cs[4]:
          if cs[4] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
          if cs[4] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[8]:
                     return True
            if cs[5] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[9]:
                  if cs[9] <= cs[3]:
                     return True
              if cs[8] <= cs[3]:
                if cs[3] <= cs[9]:
                  if cs[9] <= cs[2]:
                     return True
              if cs[8] <= cs[9]:
                if cs[9] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[9] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[5] <= cs[9]:
              if cs[9] <= cs[3]:
                if cs[3] <= cs[8]:
                  if cs[8] <= cs[2]:
                     return True
              if cs[9] <= cs[8]:
                if cs[8] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[6] <= cs[8]:
          if cs[8] <= cs[2]:
            if cs[2] <= cs[9]:
              if cs[9] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[8] <= cs[3]:
            if cs[3] <= cs[9]:
              if cs[9] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
          if cs[8] <= cs[4]:
            if cs[4] <= cs[9]:
              if cs[9] <= cs[5]:
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[8] <= cs[5]:
            if cs[5] <= cs[9]:
              if cs[9] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
          if cs[8] <= cs[9]:
            if cs[9] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[9] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
            if cs[9] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[9] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[6] <= cs[9]:
          if cs[9] <= cs[3]:
            if cs[3] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[9] <= cs[4]:
            if cs[4] <= cs[8]:
              if cs[8] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[8] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[9] <= cs[5]:
            if cs[5] <= cs[8]:
              if cs[8] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
          if cs[9] <= cs[8]:
            if cs[8] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[8] <= cs[4]:
              if cs[4] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[4] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[8] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
    if cs[1] <= cs[8]:
      if cs[8] <= cs[2]:
        if cs[2] <= cs[9]:
          if cs[9] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[8] <= cs[3]:
        if cs[3] <= cs[9]:
          if cs[9] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[8] <= cs[4]:
        if cs[4] <= cs[9]:
          if cs[9] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[9] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[5] <= cs[7]:
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
      if cs[8] <= cs[5]:
        if cs[5] <= cs[9]:
          if cs[9] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[4] <= cs[6]:
              if cs[6] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
      if cs[8] <= cs[6]:
        if cs[6] <= cs[9]:
          if cs[9] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[9] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[7] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[9] <= cs[7]:
            if cs[7] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[7] <= cs[4]:
              if cs[4] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[4] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[7] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
      if cs[8] <= cs[9]:
        if cs[9] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[9] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[9] <= cs[4]:
          if cs[4] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
          if cs[4] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[5] <= cs[7]:
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[9] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[4] <= cs[6]:
              if cs[6] <= cs[3]:
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[2]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[2]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[6] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
        if cs[9] <= cs[6]:
          if cs[6] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[6] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[7] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[7] <= cs[4]:
              if cs[4] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[4] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
                if cs[5] <= cs[3]:
                  if cs[3] <= cs[2]:
                     return True
            if cs[7] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[9] <= cs[7]:
          if cs[7] <= cs[2]:
            if cs[2] <= cs[6]:
              if cs[6] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[7] <= cs[3]:
            if cs[3] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[7] <= cs[6]:
            if cs[6] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[6] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[6] <= cs[4]:
              if cs[4] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[4] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
    if cs[1] <= cs[9]:
      if cs[9] <= cs[2]:
        if cs[2] <= cs[8]:
          if cs[8] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[9] <= cs[3]:
        if cs[3] <= cs[8]:
          if cs[8] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[2] <= cs[6]:
              if cs[6] <= cs[7]:
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[2] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
      if cs[9] <= cs[4]:
        if cs[4] <= cs[8]:
          if cs[8] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[8] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[5] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
      if cs[9] <= cs[5]:
        if cs[5] <= cs[8]:
          if cs[8] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[4] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
      if cs[9] <= cs[8]:
        if cs[8] <= cs[2]:
          if cs[2] <= cs[3]:
            if cs[3] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[3] <= cs[6]:
              if cs[6] <= cs[4]:
                if cs[4] <= cs[7]:
                  if cs[7] <= cs[5]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[3] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[8] <= cs[3]:
          if cs[3] <= cs[2]:
            if cs[2] <= cs[4]:
              if cs[4] <= cs[5]:
                if cs[5] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[2] <= cs[5]:
              if cs[5] <= cs[4]:
                if cs[4] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[2] <= cs[6]:
              if cs[6] <= cs[7]:
                if cs[7] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[2] <= cs[7]:
              if cs[7] <= cs[6]:
                if cs[6] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
        if cs[8] <= cs[4]:
          if cs[4] <= cs[2]:
            if cs[2] <= cs[5]:
              if cs[5] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
          if cs[4] <= cs[5]:
            if cs[5] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
            if cs[5] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[5] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[8] <= cs[5]:
          if cs[5] <= cs[4]:
            if cs[4] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[6]:
                  if cs[6] <= cs[7]:
                     return True
                if cs[3] <= cs[7]:
                  if cs[7] <= cs[6]:
                     return True
            if cs[4] <= cs[6]:
              if cs[6] <= cs[2]:
                if cs[2] <= cs[7]:
                  if cs[7] <= cs[3]:
                     return True
              if cs[6] <= cs[7]:
                if cs[7] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[6]:
                  if cs[6] <= cs[3]:
                     return True
              if cs[7] <= cs[6]:
                if cs[6] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        if cs[8] <= cs[6]:
          if cs[6] <= cs[2]:
            if cs[2] <= cs[7]:
              if cs[7] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[6] <= cs[3]:
            if cs[3] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
          if cs[6] <= cs[4]:
            if cs[4] <= cs[7]:
              if cs[7] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[7] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
          if cs[6] <= cs[7]:
            if cs[7] <= cs[2]:
              if cs[2] <= cs[3]:
                if cs[3] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[7] <= cs[3]:
              if cs[3] <= cs[2]:
                if cs[2] <= cs[4]:
                  if cs[4] <= cs[5]:
                     return True
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[4]:
                     return True
            if cs[7] <= cs[4]:
              if cs[4] <= cs[2]:
                if cs[2] <= cs[5]:
                  if cs[5] <= cs[3]:
                     return True
              if cs[4] <= cs[5]:
                if cs[5] <= cs[2]:
                  if cs[2] <= cs[3]:
                     return True
        
  return False
