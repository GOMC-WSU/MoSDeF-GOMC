"""General module for IO operations in MOSDEF-GOMC."""

import os

from pkg_resources import resource_filename
import importlib_resources


def get_mosdef_gomc_fn(filename):
    """Get the full path to one of the reference files shipped for utils.
    In the source distribution, these files are in ``mosdef_gomc/utils/files``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    filename : str
        Name of the file to load (with respect to the files/ folder).
    Returns
    -------
    fn : str
        Full path to filename
    """
    ref = importlib_resources.files("mosdef_gomc") / f'{"utils"}/{"files"}/{filename}'
    with importlib_resources.as_file(ref) as path:
        if not os.path.exists(path):
            raise IOError("Sorry! {} does not exists.".format(path))
        return path
