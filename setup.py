"""mosdef_gomc: The GOMC Format provider for the mosdef ecosystem.
"""
from setuptools import find_packages, setup

#####################################
NAME = "mosdef_gomc"
VERSION = "0.1.0"
ISRELEASED = True
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev0"
#####################################

if __name__ == "__main__":

    setup(
        name=NAME,
        version=__version__,
        description=__doc__.split("\n"),
        long_description=__doc__,
        author="Brad Crawford",
        author_email="",
        url="https://github.com/GOMC-WSU/MosDeF-GOMC",
        download_url="https://github.com/GOMC-WSU/MosDeF-GOMC/tarball/{}".format(
            __version__
        ),
        packages=find_packages(),
        package_dir={"mosdef_gomc": "mosdef_gomc"},
        include_package_data=True,
        license="MIT",
        zip_safe=False,
        keywords="mosdef_gomc",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )
