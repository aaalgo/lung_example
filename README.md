# lung_example

# Prerequisites:
- SimpleITK (for Luna16): http://www.simpleitk.org/SimpleITK/resources/software.html
- `git clone https://github.com/aaalgo/aardvark` so that aardvark and
  lung_example are in parallel.

# Tasks to do:
- Setup C++ code. (./setup.py build)
- Setup Luna16 data. Create symbolic link so that theres
  lung_example/luna16 which contains the following content.
```
luna16/
├── CSVFILES
│   ├── annotations.csv
│   ├── candidates.csv
│   └── sampleSubmission.csv
├── subset0
│   ├──
1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd
│   ├──
1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.raw
```
- Setup scratch directory under `lung_examle`.  It could be a simbolic
  link to some storage with at least 250G space.

- Run `./import_lung.py` to preprocess Luna16 data.
- Run `./visualize_nodules.py` to generate some sample visualization.
- Run `./train.py` to start training.  No need to finish.  Kill it after
  the progress bar starts moving.  
