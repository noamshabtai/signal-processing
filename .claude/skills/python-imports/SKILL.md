---
name: python-imports
description: Import-style convention for this repo. Apply whenever writing, editing, or reviewing Python imports — before adding any `import` or `from ... import` statement.
---

# Python import style

- Always use `import module` style. Never use `from module import name`.
- The only exception is local intra-package imports: `from . import sibling_module`.
- Do not alias modules with `import X as Y`, except for the two community
  conventions `import numpy as np` and `import matplotlib.pyplot as plt`.
