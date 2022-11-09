# Installation

Run in the repository root:
> pip install -e .

Also install the dependencies in blenderproc:
> python scripts/install_blenderproc.py

 


### Blenderproc Debugging
To make Debugging work:  
`blenderproc pip install pydevd-pycharm~=PYCHARM_VERSION`

### Install bpy in IDE conda  

UPDATE: We do not need to install `bpy`.
Instead use the `fake-bpy-module-BPY_VERSION` module.

To make `bpy` work:  
 - Use Python 3.7!  
 - `pip install future-fstrings`  
 - `sudo apt install subversion`
 - Prebuilt wheels: https://drive.google.com/drive/folders/18HFAbqoPBF6ItYrwbeQB-xSEbq3VudsI install with `pip install`
 - Preview wheel for 2.91 if nothing works: https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0