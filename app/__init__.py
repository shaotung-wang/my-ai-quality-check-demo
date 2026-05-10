"""
app package: logical grouping of the existing modules.

This package provides lightweight wrappers that re-export the current
top-level modules under a more structured namespace (e.g. app.inference,
app.workers, app.gui, app.tools). The original modules remain at the
top-level to keep backward compatibility.
"""
__all__ = [
    'inference',
    'workers',
    'gui',
    'tools',
    'config',
]

