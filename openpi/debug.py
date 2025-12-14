import h5py
f = h5py.File('/home/stella/projects/vggt/libero/datasets_with_vggt/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5', 'r')
print(list(f['data/demo_0'].keys()))  # See what keys exist