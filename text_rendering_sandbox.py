# %%

import logging

from framework import Context
from framework import logger
from framework import set_context, line

logger().setLevel(logging.INFO)

size = (500, 500)

context = Context(size)
set_context(context)

# %% 
l1 = line((250, 100), (50, 450))
l2 = line((250, 100), (450, 450))
l3 = line((100, 250), (400, 250))

l1.union(l2).union(l3).fill(inflate=5, fg_color=(1, 1, 1, 1), bg_color=(0, 0, 0, 1)).to_rgb().show()
# .fill(inflate=5, fg_color=(1.0, 1.0, 1.0, 1.0), bg_color=(0,0,0,1)).to_rgb().show()
# %%
