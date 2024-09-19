Combs2: Convergent motifs for binding sites 2.0
-----------------------------------
SYNOPSIS
+++++++++

Combs2 is an open-source Python package for interfacial protein design.  
Combs generates and uses van der Mers (vdMs) to analyze and design 
interaction motifs in proteins, in small-molecule-protein interfaces as 
well as larger protein-protein interfaces.  

INSTALLATION
++++++++++++
Install Combs2 by cloning the github repository, e.g. ::

    > git clone https://github.com/npolizzi/Combs2024.git

Note that you can keep Combs2 up-to-date (useful for bug-fixes and updates to the code) by: ::

    > git pull https://github.com/npolizzi/Combs2024.git

Within Combs2/docs/ use the **env_combs_(platform).yml** file to create a conda virtual environment for Combs2 by, e.g. ::

    > conda env create -f env_combs_mac.yml

You will need to update the `prefix:` line at the bottom of the .yml file to point to your miniconda installation directory. 
Change the path up to .../miniconda3/ the environment directories will be created for you ::

This will install all the necessary packages on which Combs2 depends but requires miniconda to be pre-installed. 
To install miniconda for Mac, go to the `Anaconda website <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_ and follow the instructions.

If you are experiencing problems installing the environment, a minimal conda environment that should get you going can be created by ::

    > conda create --name env_combs python=3.8
    > conda activate env_combs
    > pip install sklearn scipy pandas numba numpy prody pyarrow jupyter matplotlib

If you are installing on a Mac M1, numba is not yet pip-installable. Try this workaround: ::

    > pip install -i https://pypi.anaconda.org/numba/label/wheels_experimental_m1/simple llvmlite
    > pip install "numpy==1.21.*" importlib-metadata
    > pip install -i https://pypi.anaconda.org/numba/label/wheels_experimental_m1/simple numba

Note that, for some older Mac operating systems, you may need to specify scipy==1.1.0 (If you see a Segmentation Fault when trying to use Combs2, try this.). After creating the env_combs environment, add the path to Combs2 to your **.bash_profile** or **.bashrc** (which is usually in your home directory): ::

    > vi .bash_profile
    export PYTHONPATH="/Users/npolizzi/Projects/design/Combs2/:$PYTHONPATH"

Exchange the python path above to your own path to Combs2.  For Linux, you may need to drop the quotes or change them to single quotes.  Now you can activate the Combs2 virtual environment and use Combs2, interactively or in scripts, e.g. ::

    > conda activate env_combs
    > ipython

Then, in an IPython session or Juptyer notebook:

.. code-block:: python

    import combs2

Note that you should activate the conda environment before using Combs.  Otherwise you will likely encounter missing-package errors.

DATABASE
++++++++

The van der Mer databases are currently hosted on DropBox (subject to change) at this `link <https://www.dropbox.com/sh/a5wakk7nonc03bv/AACbar6bDBua-HH7L_-2iO-0a?dl=0>`_.  The zipped file is about 7.5 GB currently.  For a list of available chemical groups and the amino acids from which they derive, see **CG_dicts.txt** in Combs2/combs2/files/.  Download and unzip the database file by: ::

    > tar -xvf database.tar.gz

You do not need to unzip the gzipped files within the vdMs folder.

SUPPORTING SOFTWARE
+++++++++++++++++++
Some of the Combs functionality (e.g. classifying rotamers or secondary structure) currently depends on third-party software such as DSSP and MolProbity. You are encouraged to download and `install DSSP <https://swift.cmbi.umcn.nl/gv/dssp/>`_ (and also point ProDy to the DSSP installation path if necessary).  You are highly encouraged to download and install MolProbity from the Richardson Lab's `GitHub page <https://github.com/rlabduke/MolProbity>`_.  Some functions within Combs require a path to Probe, Reduce, or Rotalyze within MolProbity.   

TUTORIAL
++++++++
*Update:* See the script **Combs2/tutorials/HPC_scripts/biotin/run_design.py** for an example of the most up-to-date usage of COMBS.

DOCUMENTATION
+++++++++++++
Very little unfortunately, see tutorial and example scripts.

LICENSE
+++++++

Combs2 is available under MIT License. See LICENSE.txt for more details.
